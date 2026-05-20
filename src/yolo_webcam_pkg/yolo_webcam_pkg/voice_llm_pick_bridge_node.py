#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import queue
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from yolo_webcam_pkg.llm_command_parser import llm_chat_or_pick


INSTALL_HELP = 'pip install sounddevice webrtcvad faster-whisper numpy'

STOP_KEYWORDS = [
    '멈춰',
    '멈추',
    '정지',
    '중지',
    '스톱',
    '스탑',
    'stop',
    'emergency stop',
]


def default_whisper_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


class VoiceLLMPickBridgeNode(Node):
    def __init__(self):
        super().__init__('voice_llm_pick_bridge_node')

        device_default = default_whisper_device()
        compute_type_default = 'float16' if device_default == 'cuda' else 'int8'

        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('vad_mode', 2)
        self.declare_parameter('frame_ms', 30)
        self.declare_parameter('silence_end_ms', 800)
        self.declare_parameter('min_speech_ms', 400)
        self.declare_parameter('max_speech_sec', 8.0)
        self.declare_parameter('whisper_model', 'base')
        self.declare_parameter('whisper_device', device_default)
        self.declare_parameter('whisper_compute_type', compute_type_default)
        self.declare_parameter('language', 'ko')
        self.declare_parameter('topic_name', '/llm/target_pick')
        self.declare_parameter('confirm_text', False)
        self.declare_parameter('cooldown_sec', 1.0)

        self.sample_rate = int(self.get_parameter('sample_rate').value)
        self.vad_mode = int(self.get_parameter('vad_mode').value)
        self.frame_ms = int(self.get_parameter('frame_ms').value)
        self.silence_end_ms = int(self.get_parameter('silence_end_ms').value)
        self.min_speech_ms = int(self.get_parameter('min_speech_ms').value)
        self.max_speech_sec = float(self.get_parameter('max_speech_sec').value)
        self.whisper_model_name = str(self.get_parameter('whisper_model').value)
        self.whisper_device = str(self.get_parameter('whisper_device').value)
        self.whisper_compute_type = str(self.get_parameter('whisper_compute_type').value)
        self.language = str(self.get_parameter('language').value)
        self.topic_name = str(self.get_parameter('topic_name').value)
        self.confirm_text = bool(self.get_parameter('confirm_text').value)
        self.cooldown_sec = float(self.get_parameter('cooldown_sec').value)

        if self.frame_ms not in (10, 20, 30):
            self.get_logger().warn(
                f'frame_ms={self.frame_ms} is invalid for webrtcvad. Using 30 ms.'
            )
            self.frame_ms = 30

        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self.stop_event = threading.Event()
        self.audio_thread: threading.Thread | None = None
        self.pending_pcm = bytearray()
        self.recording = False
        self.audio_buffer = bytearray()
        self.speech_ms = 0
        self.silence_ms = 0
        self.recording_ms = 0
        self.last_text = ''
        self.last_text_time = 0.0
        self.last_publish_time = 0.0

        self.visible_objects = {
            'blue_cube',
            'blue_cylinder',
            'objects-yellow_box',
            'yellow_cylinder',
        }
        self.pick_pub = self.create_publisher(String, self.topic_name, 10)
        self.stop_pub = self.create_publisher(String, '/llm/stop', 10)

        self.get_logger().info('Voice bridge started')

        try:
            self.sounddevice, self.webrtcvad, whisper_model_cls = self.load_voice_dependencies()
        except ImportError as exc:
            self.get_logger().error(f'Voice dependencies are missing: {exc}')
            self.get_logger().error(f'Install with: {INSTALL_HELP}')
            return

        if self.sample_rate != 16000:
            self.get_logger().error(
                'This node expects 16 kHz mono int16 PCM for webrtcvad. '
                f'sample_rate={self.sample_rate} is not supported.'
            )
            return

        self.vad = self.webrtcvad.Vad(self.vad_mode)
        self.get_logger().info(
            f'Loading Whisper model={self.whisper_model_name} '
            f'device={self.whisper_device} compute_type={self.whisper_compute_type}'
        )
        try:
            self.whisper_model = whisper_model_cls(
                self.whisper_model_name,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to load Whisper model: {exc}')
            self.get_logger().error(f'Install/check runtime with: {INSTALL_HELP}')
            return

        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()

    @staticmethod
    def load_voice_dependencies():
        import sounddevice
        import webrtcvad
        from faster_whisper import WhisperModel

        return sounddevice, webrtcvad, WhisperModel

    def audio_callback(self, indata, frames, time_info, status):
        del frames, time_info
        if status:
            self.get_logger().warn(f'Microphone status: {status}')

        pcm = np.asarray(indata, dtype=np.int16).reshape(-1).tobytes()
        try:
            self.audio_queue.put_nowait(pcm)
        except queue.Full:
            pass

    def audio_loop(self):
        try:
            with self.sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.frame_samples,
                callback=self.audio_callback,
            ):
                self.get_logger().info('Listening microphone...')
                while rclpy.ok() and not self.stop_event.is_set():
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    self.pending_pcm.extend(data)
                    self.process_pending_frames()
        except Exception as exc:
            self.get_logger().error(f'Microphone loop failed: {exc}')
            self.get_logger().error(f'Install/check audio runtime with: {INSTALL_HELP}')

    def process_pending_frames(self):
        while len(self.pending_pcm) >= self.frame_bytes:
            frame = bytes(self.pending_pcm[:self.frame_bytes])
            del self.pending_pcm[:self.frame_bytes]
            self.process_vad_frame(frame)

    def process_vad_frame(self, frame: bytes):
        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
        except Exception as exc:
            self.get_logger().warn(f'VAD failed for audio frame: {exc}')
            return

        if not self.recording:
            if not is_speech:
                return
            self.recording = True
            self.audio_buffer = bytearray()
            self.speech_ms = 0
            self.silence_ms = 0
            self.recording_ms = 0
            self.get_logger().info('Speech started')

        self.audio_buffer.extend(frame)
        self.recording_ms += self.frame_ms

        if is_speech:
            self.speech_ms += self.frame_ms
            self.silence_ms = 0
        else:
            self.silence_ms += self.frame_ms

        if self.recording_ms >= int(self.max_speech_sec * 1000):
            self.finish_utterance()
            return

        if self.silence_ms >= self.silence_end_ms:
            self.finish_utterance()

    def finish_utterance(self):
        audio_pcm = bytes(self.audio_buffer)
        speech_ms = self.speech_ms
        self.recording = False
        self.audio_buffer = bytearray()
        self.speech_ms = 0
        self.silence_ms = 0
        self.recording_ms = 0

        if speech_ms < self.min_speech_ms:
            self.get_logger().info(
                f'Ignored short speech segment: speech_ms={speech_ms}'
            )
            return

        self.get_logger().info('Speech ended. Transcribing...')
        self.handle_utterance(audio_pcm)

    def handle_utterance(self, audio_pcm: bytes):
        audio_i16 = np.frombuffer(audio_pcm, dtype=np.int16)
        if audio_i16.size == 0:
            return
        audio = audio_i16.astype(np.float32) / 32768.0

        try:
            segments, _info = self.whisper_model.transcribe(
                audio,
                language=self.language or None,
                beam_size=1,
                vad_filter=False,
            )
            text = ' '.join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            self.get_logger().error(f'Whisper transcription failed: {exc}')
            return

        if not text:
            self.get_logger().info('Ignored non-pick command: empty transcription')
            return

        self.get_logger().info(f'STT text: {text}')

        if self.is_stop_command(text):
            self.publish_stop_command(text)
            return

        now = time.monotonic()
        if text == self.last_text and now - self.last_text_time < self.cooldown_sec:
            self.get_logger().info(f'Ignored duplicate STT text within cooldown: {text}')
            return
        self.last_text = text
        self.last_text_time = now

        result = llm_chat_or_pick(text, visible_objects=self.visible_objects)
        self.get_logger().info(
            f'Parsed command: {json.dumps(result, ensure_ascii=False)}'
        )

        if result.get('type') != 'pick_queue':
            self.get_logger().info(f'Ignored non-pick command: {result.get("reply", text)}')
            return

        if now - self.last_publish_time < self.cooldown_sec:
            self.get_logger().info('Ignored pick command during publish cooldown.')
            return

        if self.confirm_text:
            self.get_logger().info(
                'confirm_text=true; parsed pick command was not auto-published.'
            )
            return

        msg_data = self.build_target_message(result)
        if msg_data is None:
            return

        msg = String()
        msg.data = json.dumps(msg_data, ensure_ascii=False)
        self.pick_pub.publish(msg)
        self.last_publish_time = time.monotonic()
        self.get_logger().info(f'Published /llm/target_pick: {msg.data}')

    def build_target_message(self, result: dict[str, Any]) -> dict[str, Any] | None:
        labels = result.get('labels', [])
        if not labels:
            self.get_logger().warn('pick_queue result has no labels.')
            return None
        if len(labels) != 1:
            self.get_logger().warn(
                f'Expected exactly one label, got {labels}. Not publishing.'
            )
            return None

        label = labels[0]
        msg_data: dict[str, Any] = {
            'command_id': time.time_ns(),
            'label': label,
            'labels': labels,
        }

        place_mode = result.get('place_mode')
        if place_mode == 'relative':
            msg_data['place_mode'] = place_mode
            msg_data['place_dx'] = float(result.get('place_dx', 0.0))
            msg_data['place_dy'] = float(result.get('place_dy', 0.0))
            msg_data['place_dz'] = float(result.get('place_dz', 0.0))
        elif place_mode == 'zone':
            msg_data['place_mode'] = place_mode
            msg_data['zone'] = str(result.get('zone', '')).upper()

        return msg_data

    @staticmethod
    def is_stop_command(text: str) -> bool:
        normalized = (text or '').strip().lower()
        compact = ''.join(normalized.split())
        return any(
            keyword in normalized or ''.join(keyword.split()) in compact
            for keyword in STOP_KEYWORDS
        )

    def publish_stop_command(self, text: str):
        self.get_logger().info(f'Voice stop command detected: {text}')
        msg_data = {
            'command_id': time.time_ns(),
            'type': 'stop',
            'text': text,
        }
        msg = String()
        msg.data = json.dumps(msg_data, ensure_ascii=False)
        self.stop_pub.publish(msg)
        self.get_logger().info(f'Published /llm/stop: {msg.data}')

    def destroy_node(self):
        self.stop_event.set()
        if self.audio_thread is not None and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceLLMPickBridgeNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
