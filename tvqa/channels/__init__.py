# -*- coding: utf-8 -*-
"""系统信号通道层：串口 + ADB 工厂。"""

from .adb_ch import AdbChannel, MockAdbChannel, RealAdbChannel, create_adb_channel
from .serial_ch import MockSerialChannel, PyserialChannel, SerialChannel, create_serial_channel

__all__ = ["SerialChannel", "PyserialChannel", "MockSerialChannel", "create_serial_channel",
           "AdbChannel", "RealAdbChannel", "MockAdbChannel", "create_adb_channel"]
