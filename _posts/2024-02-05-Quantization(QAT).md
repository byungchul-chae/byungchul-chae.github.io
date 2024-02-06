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

## 1. Introduction

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

## 2. Quantization Inference

### 2.1. Quantization Scheme

신경망에서의 추론 과정을 효율적으로 만들기 위해, 우리는 특별한 양자화 방식을 사용한다. 이 방식은 신경망이 학습할 때 사용되는 부동 소수점 수치를 정수 형태로 변환함으로써, 추론 시 오직 정수 연산만을 사용하게 한다. 이렇게 하면 복잡한 부동 소수점 연산을 피할 수 있어, 모바일 장치와 같이 제한된 계산 능력을 가진 환경에서도 빠르고 효율적인 추론이 가능하다.

양자화된 값(\(q\))과 실제 수치 값(\(r\)) 사이의 관계를 정의하는 공식은 \(r = S(q - Z)\)이다. 여기서 \(S\)는 스케일(scale)을, \(Z\)는 제로 포인트(zero-point)를 나타낸다. 이 두 매개변수는 양자화 과정에서 중요한 역할을 한다.

- **스케일(Scale):** 실제 값과 양자화된 값 사이의 비율을 정의한다. 이 값은 양자화 과정에서 값의 범위를 조정하는 데 사용된다.
- **제로 포인트(Zero-Point):** 실제 0 값이 양자화된 값에서 어떻게 표현되는지를 나타낸다. 이를 통해 0을 포함한 모든 실제 값이 정확하게 양자화될 수 있도록 보장한다.

#### 양자화 과정의 중요성

- **정수 연산만 사용:** 추론 과정에서 부동 소수점 연산을 사용하지 않고, 정수 연산만을 사용함으로써 계산 속도를 향상시킨다. 이는 특히 SIMD(단일 명령 다중 데이터) 하드웨어에서 더욱 효과적이다.
- **제로 패딩 지원:** 신경망 연산에서는 종종 배열의 경계 주변에 0으로 채우는 제로 패딩이 필요하다. 제로 포인트를 통해 0 값이 정확히 표현되므로, 이러한 연산을 효율적으로 수행할 수 있다.

#### 양자화된 버퍼 데이터 구조

양자화된 값을 저장하고 관리하기 위해, 우리는 `QuantizedBuffer`라는 특별한 데이터 구조를 사용한다. 이 구조는 다음과 같은 구성 요소를 가진다:

- **q:** 양자화된 값을 저장하는 벡터. 이 값들은 정수 형태로 저장된다.
- **S:** 스케일 값. 실제 값과 양자화된 값 사이의 비율을 나타낸다.
- **Z:** 제로 포인트. 실제 0 값이 양자화된 값에서 어떻게 표현되는지를 나타낸다.

이 데이터 구조를 사용함으로써, 각 활성화 배열과 가중치 배열에 대해 일관된 양자화 매개변수를 적용할 수 있으며, 신경망의 모든 부분에서 효율적인 정수 연산을 지원한다.

### 결론

양자화 스킴은 신경망의 추론 속도를 높이는 데 중요한 역할을 한다. 이 방식을 통해, 복잡한 부동 소수점 연산을 피하고, 제한된 계산 능력을 가진 장치에서도 빠른 추론이 가능하다. 이러한 접근 방식은 특히 모바일 장치나 임베디드 시스템에서 신경망을 실행할 때 중요하다.

### 2.2. Integer-arithmetic-only matrix multiplication
### 2.3. Efficient handling of zero-points
### 2.4. Implementation of a typical fused layer
## 3. Training with simulated quantization
### 3.1. Learning quantization ranges
### 3.2. Batch normalization folding
## 4. Experiments
### 4.1. Quantization training of Large Networks
#### 4.1.1. ResNets
#### 4.1.2. Inception v3 on ImageNet
### 4.2. Quantization of MobileNets
#### 4.2.1. ImageNet
#### 4.2.2. COCO
#### 4.2.3. Face detection and attribute classification
## 5. Discussion
