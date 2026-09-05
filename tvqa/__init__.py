# -*- coding: utf-8 -*-
"""tvqa：电视音画质量自动化检测工程。

模块布局（与《电视音画质量自动化检测方案.md》对应）：
- config / logging_setup / clock / utils   基础设施
- sources/    输入抽象（帧目录、视频文件、音频文件、采集卡）
- channels/   系统信号通道抽象（串口、ADB，真实 + mock 双实现）
- detectors/  检测器（luma 黑白闪、stutter、corruption、avsync）
- probe / correlate                        系统信号采集与跨通道归因
- archive / report                         归档追溯与 HTML 报告
- evaluate/                                数据集真值匹配与记分卡
- scenarios                                演示/故障注入场景驱动
"""

__version__ = "0.1.0"
