---
title:  "[Paper Review] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only-Inference"
excerpt: "[Paper Review] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only-Inference"
categories: [Paper]
tags: [quantization, qat, cvpr, paper review]

toc: true
toc_sticky: true
 
date: 2024-02-05
last_modified_at: 2024-02-05 
---

## Info
Tensorflow에서 제공하는 QAT 관련 논문 요약

## Introduction

최신 컨볼루션 신경망(CNN)을 모바일 플랫폼에 배포하는 방법은 크게 두 가지 범주로 나뉜다:

1. **계산 및 메모리 효율성을 강조한 새로운 네트워크 아키텍처 설계:** 
2. **CNN의 가중치 및 활성화를 32비트 부동 소수점에서 낮은 비트 심도로 양자화:** 

양자화 접근법은 모델 크기와 추론 시간을 줄이는 데 중점을 두고 있지만, 기존 방법들은 두 가지 주요 문제에 직면해 있다:

- **과다 매개변수화(Over-parameterization):** AlexNet, VGG, GoogleNet과 같은 기존 아키텍처들은 정확도를 소폭 향상시키기 위해 과다하게 매개변수화되어 있어, 양자화 실험 결과가 실질적인 의미를 갖기 어려움.
- **하드웨어 효율성 부족:** 많은 양자화 접근법들이 실제 하드웨어에서의 효율성 개선을 입증하지 못함. 특히, 가중치만을 양자화하는 방법은 저장 공간 절약에는 도움이 되지만, 계산 효율성은 크게 개선하지 못함.

### Contributions

이 논문은 모바일 CPU에서의 추론 속도와 정확도 사이의 균형을 개선하기 위한 양자화 체계와 관련된 다음과 같은 주요 기여를 제공한다:

1. **양자화 방식:** 가중치와 활성화를 8비트 정수로 양자화하고, 편향 벡터에 대해서만 32비트 정수를 사용. 모델의 정확도 손실을 최소화하면서 모바일 장치에서의 실행 속도를 향상시키는 데 중점을 둠.

2. **정수 산술 전용 하드웨어에서의 효율적인 추론 프레임워크:** Qualcomm Hexagon과 같은 특정 하드웨어에서 효율적으로 구현할 수 있는 양자화된 추론 프레임워크를 제공.

3. **개선된 벤치마크 결과:** 이 프레임워크를 MobileNets와 같은 효율적인 분류 및 탐지 시스템에 적용하여, 인기 있는 ARM CPU에서의 ImageNet 분류와 COCO 객체 탐지를 포함한 작업에서의 지연 시간 대비 정확도 트레이드오프에서 중요한 개선을 보여줌.

## Quantization Inference

### Quantization Scheme
### Integer-arithmetic-only matrix multiplication
### Efficient handling of zero-points
### Implementation of a typical fused layer
## Training with simulated quantization
### Learning quantization ranges
### Batch normalization folding
## Experiments
### Quantization training of Large Networks
#### ResNets
#### Inception v3 on ImageNet
### Quantization of MobileNets
#### ImageNet
#### COCO
#### Face detection and attribute classification
## Discussion
