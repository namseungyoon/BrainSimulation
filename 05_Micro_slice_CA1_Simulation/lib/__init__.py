# -*- coding: utf-8 -*-
"""lib — micro-slice CA1 재사용 모듈 (번호 없는 유일한 import 통로).

교차 카테고리(01_tissue/02_neurons/03_network/04_experiments) 재사용 로직은
전부 여기에 둔다. 단계 폴더의 실행 스크립트는 번호로 시작해 import 불가하므로,
재사용은 반드시 이 패키지(+ ../shared/common)를 통한다.

계획 모듈(진행 시 작성):
  microslice_io · atlas_geom · morph_transform · model_registry
  connectome_rules · synapse_params · mea_forward
"""
