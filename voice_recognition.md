#### 원문 링크 : https://www.kaggle.com/davids1992/speech-representation-and-data-exploration

# MFCC 개요 
What is MFCC? MFCC는 오디오 신호에서 추출할 수 있는 feature로, 소리의 고유한 특징을 나타내는 수치입니다. 주로 음성 인식, 화자 인식, 음성 합성, 음악 장르 분류 등 오디오 도메인의 문제를 해결하는 데 사용됩니다.

## MFCC 과정
1. 기존 신호를 짧게 쪼개기 : 오디오 신호는 끝없이 변하므로 분석의 편의성을 위해 짧게 조개는 것이 필요
2. FFT(고속푸리에변환) 적용 : 기존의 데이터가 x축(시간) y축(음압)이었다면, x축(주파수) y축(음압)으로 변경 why? 주파수 정보가 있어야 고유한 소리의 특징 추출 가능. FFT를 통해서 나온 데이터가 Spectrum
3. Mel Scale 기반 Filter Bank를 Spectrum에 적용 : 인간의 귀는 저주파수와 고주파수를 다르게 처리하는 데, 물리적 주파수와 사람이 실제 인식하는 주파수의 관계를 표현한 것이 Mel Scale임. 이렇게 해서 나온 데이터가 Mel Spectrum
4. Mel Spectrum에 Ceptral 분석 진행 : 소리의 고유한 특징은 배음 구조에서 비롯되는데 그 배음 구조를 추출하는 분석이 Ceptral Analysis. 이 과정에서 IFFT(역 고속푸리에변환)이 사용됨.

#### 관련 링크 : https://brightwon.tistory.com/11

## What is 푸리에 변환?
푸리에 변환(Fourier transform)을 직관적으로 설명하면 푸리에 변환은 임의의 입력 신호를 다양한 주파수를 갖는 주기함수들의 합으로 분해하여 표현하는 것이다.
푸리에 변환(Fourier transform)의 대단한 점은 입력 신호가 어떤 신호이든지 관계없이 임의의 입력 신호를 sin, cos 주기함수들의 합으로 항상 분해할 수 있다는 것이다. 그리고 그 과정을 수식으로 표현한 것이 푸리에 변환식이다.

#### 관련 링크 : https://darkpgmr.tistory.com/171
