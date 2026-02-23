# =============================================================================
# AUDIO PLAYBACK — callback-driven OutputStream
# =============================================================================
#
# PortAudio (via sounddevice) offers two ways to play audio:
#
# Blocking  sd.play(array) / sd.OutputStream.write(block)
#   Hand a complete array to PortAudio and it handles the rest.
#   Simple, but you must have all the audio before you start — incompatible
#   with streaming TTS where chunks arrive one at a time from the model.
#
# Callback  sd.OutputStream(callback=fn)
#   PortAudio runs a real-time audio thread that calls fn() every blocksize
#   frames to pull the next chunk of audio.  Rules: no allocation, no
#   blocking I/O, no Python locks — any delay causes an audible glitch.
#   A queue bridges the real-time audio thread and the synthesis thread.
#
# This module uses the callback approach so synthesis and playback overlap:
# the stream starts on the first chunk and plays continuously while the model
# is still generating the rest of the sentence.
#
# ALTERNATIVES FOR THE QUEUE / BUFFER DESIGN
# -------------------------------------------
# queue.Queue (thread-safe, blocking get)
#   get_nowait() still works, but Queue adds unnecessary locking overhead for
#   a single-producer / single-consumer case.  SimpleQueue is lighter.
#
# Pre-allocated ring buffer
#   Avoids the per-block np.zeros allocation in _push_block.  More complex
#   to implement; worthwhile only if synthesis is extremely latency-sensitive.
#
# Collecting all audio first, then sd.play()
#   Zero complexity — call sd.play(all_audio) after the generator is exhausted.
#   First audio is delayed by the full synthesis time; unacceptable for long
#   sentences but perfectly fine for short fixed responses.
#
# asyncio + asyncio.Queue
#   Replaces the threading model entirely.  Works well if the rest of the
#   application is async (e.g. an LLM streaming loop).  The PortAudio callback
#   itself is always synchronous regardless of the outer async framework.
# =============================================================================

import queue
from threading import Event

import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self, sample_rate: int = 24_000, blocksize: int = 2048) -> None:
        self.sample_rate = sample_rate
        # blocksize is the number of frames PortAudio requests per callback.
        # Smaller = lower latency but more callback overhead and glitch risk.
        # Larger = more buffering but smoother playback.  2048 ≈ 85 ms at 24 kHz.
        self.blocksize = blocksize
        # SimpleQueue is safe for single-producer / single-consumer use without
        # the extra locking overhead of queue.Queue.
        self._queue: queue.SimpleQueue[np.ndarray | None] = queue.SimpleQueue()
        # TTS chunks are variable length and rarely align to blocksize.
        # _leftover holds the tail of the previous chunk until the next call.
        self._leftover: np.ndarray | None = None
        self._done = Event()
        self._stream: sd.OutputStream | None = None

    def _callback(
        self, outdata: np.ndarray, frames: int, _time: object, _status: object
    ) -> None:
        """Fill outdata from the block queue; stop on sentinel, silence on underrun."""
        # PortAudio calls this from its own real-time thread — keep it fast.
        try:
            block = self._queue.get_nowait()  # raises Empty if nothing is ready yet
        except queue.Empty:
            # Underrun: synthesis is slower than playback at this moment.
            # Fill with silence; PortAudio will call us again next interval.
            outdata.fill(0)
            return
        if block is None:
            # None is the sentinel enqueued by stop() — all audio has been played.
            outdata.fill(0)
            raise sd.CallbackStop  # triggers finished_callback → _done.set()
        # outdata shape is (frames, channels); write mono audio into channel 0.
        outdata[:, 0] = block

    def _push_block(self, block: np.ndarray) -> None:
        """Zero-pad a partial block to blocksize and enqueue it."""
        # The callback always reads exactly blocksize samples from the queue.
        # Padding ensures the callback can copy directly into outdata without
        # a bounds check, and the silence tail is inaudible.
        padded = np.zeros(self.blocksize, dtype=np.float32)
        padded[: len(block)] = block
        self._queue.put(padded)

    def queue_audio(self, samples: np.ndarray) -> None:
        """Chunk samples into fixed blocks and start stream on first call."""
        samples = np.asarray(samples, dtype=np.float32)
        if self._leftover is not None:
            # A previous chunk ended mid-block.  Prepend the leftover so the
            # blocks stay blocksize-aligned across chunk boundaries.
            samples = np.concatenate([self._leftover, samples])
            self._leftover = None
        for i in range(0, len(samples), self.blocksize):
            block = samples[i : i + self.blocksize]
            if len(block) < self.blocksize:
                # Incomplete block — hold it until the next chunk arrives or
                # stop() is called, whichever comes first.
                self._leftover = block
            else:
                self._queue.put(block)
        if self._stream is None:
            # Lazy stream creation: start the stream on the first chunk so
            # synthesis and playback overlap from the very beginning.
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._callback,
                blocksize=self.blocksize,
                finished_callback=self._done.set,
            )
            self._stream.start()

    def stop(self) -> None:
        """Flush remaining audio, wait for playback to finish, and close stream."""
        if self._stream is None:
            return
        if self._leftover is not None:
            # Flush the final partial block with zero-padding before signalling done.
            self._push_block(self._leftover)
            self._leftover = None
        # None acts as a sentinel: the callback raises CallbackStop when it dequeues it,
        # which fires finished_callback → _done.set(), unblocking the wait below.
        self._queue.put(None)
        self._done.wait()  # block until PortAudio confirms the stream has stopped
        self._stream.close()
        self._stream = None
