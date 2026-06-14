# BraTS 2.5D: Efficient 3D Brain Tumor Segmentation via Multi-View Ensemble
 
> 
>
> 임상적 필요에 기반한 3-region 뇌종양 분할(WT/TC/ET)을 GPU 메모리 제약 속에서 2.5D multi-view 앙상블로 구현하고, XAI와 환자 클러스터링으로 의료 AI의 해석 가능성을 제시한 프로젝트입니다.
---
 
## 🎯 전체 개요
 
### 1. 문제 정의: Segmentation
- 임상적 필요성에 맞춰 **3-region segmentation(WT/TC/ET)** 으로 자발적 확장
- BraTS 2020 dataset 기반, 369명 환자 데이터 활용
### 2. 논문 기반 아키텍처 설계
- SOTA 리더보드 분석 → MICCAI 논문 검토 → 2.5D multi-view 전략 도입
- 참고 논문: Rajput et al. MVBTS (2023), Wang et al. TransBTS (MICCAI 2021), nnU-Net (MICCAI 2021)
- 6가지 모델 단계적 비교: 2D U-Net → U-Net+LSTM → U-Net++ → Attention U-Net++ → TransUNet → 3-view Ensemble
### 3. GPU 메모리 제약의 창의적 극복
- 3D U-Net 대신 **adjacent 5 slices의 2.5D 입력** (20 channels)으로 메모리 효율화
- Test-time ensemble: axial + coronal + sagittal 확률 맵 평균으로 3D 공간 정보 복원
- Triplanar ensemble 논문 기반: 2D 모델 3개의 조합이 3D 모델에 준하는 성능 달성
### 4. 의료 AI의 설명 가능성 강화
- **GradCAM++** 기반 모델 판단 근거 시각화
- 채널별/공간별 attention map으로 모델이 실제 종양 영역에 집중하는지 검증
- XAI 결과와 성능 메트릭 연결
### 5. 임상적 인사이트: 환자 유형화
- 종양 크기/위치/모델 성능을 결합한 환자 클러스터링 (GMM)
- 어려운 환자군(ET=0, 작은 종양) 식별 → 향후 개선 방향 도출

--- 
> 전체 내용은 발표자료를 통해 확인하실 수 있습니다. 
