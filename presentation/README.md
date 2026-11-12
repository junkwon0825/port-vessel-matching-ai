# Portfolio Presentation

[Port Vessel Matching AI 포트폴리오 PDF 열기](Port_Vessel_Matching_AI_Portfolio.pdf)

대회의 문제 정의, 검증 설계, 후보 검색, Learning-to-Rank 모델과 로컬 실험 결과를 12페이지로 정리한 발표자료입니다.

## 주요 내용

- 선박 ID가 겹치지 않는 vessel-disjoint validation
- 78,595척 카탈로그를 최대 600척으로 줄이는 후보 검색
- 40개 pair feature와 LightGBM LambdaRank
- 한글과 비한글 기록을 분리한 language hybrid 전략
- 코드 제출 환경과 재현성 대응

발표자료의 `0.9951`과 `0.6811`은 로컬 홀드아웃 결과입니다. 실제 대회 리더보드 점수와 구분해서 사용합니다.
