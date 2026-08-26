import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 로컬 Windows와 Streamlit Linux 배포 환경 모두 대응
if os.name == "nt":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(
        fname=font_path
    ).get_name()

    plt.rcParams["font.family"] = font_name

plt.rcParams["axes.unicode_minus"] = False

# CSS
st.markdown("""
<style>

/* 전체 화면 여백 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* 제목 */
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

/* 설명 */
.main-description {
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 2rem;
}

/* 결과 위험등급 */
.risk-badge {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 999px;
    font-size: 1.2rem;
    font-weight: 700;
    text-align: center;
    min-width: 90px;
}

.risk-badge.low {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.risk-badge.medium {
    background-color: #fef3c7;
    color: #b45309;
    border: 1px solid #fde68a;
}

.risk-badge.high {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

.result-label {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 0.2rem;
}

.result-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}

.section-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">호텔 예약 취소 위험 예측</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-description">
        예약 정보를 입력하면 머신러닝 모델이 취소 가능성과 예상 손실 위험을 분석합니다.
    </div>
    """,
    unsafe_allow_html=True
)


# 자료 불러오기
model = joblib.load("model/hotel_canceled.pkl")
risk_thresholds = joblib.load("model/risk_thresholds.pkl")

st.set_page_config(
    page_title="호텔 예약 취소 위험 예측",
    page_icon="🏨",
    layout="wide"
)

@st.cache_data
def load_dashboard_data():

    model_results = pd.read_csv("model_results.csv")
    rf_results = pd.read_csv("rf_overfitting_results.csv")
    loss_risk = pd.read_csv("validation_loss_risk.csv")
    importance_df = pd.read_csv("permutation_importance.csv")


    return model_results, rf_results, loss_risk, importance_df

model_results, rf_results, loss_risk, importance_df = load_dashboard_data()


# 범주형 변수 목록
categorical_cols = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "reserved_room_type",
    "customer_type",
    "country_group"
]

# 학습된 OneHotEncoder에서 실제 카테고리 가져오기
encoder = (
    model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

category_options = {
    col: list(categories)
    for col, categories in zip(
        categorical_cols,
        encoder.categories_
    )
}


# 사용자 입력 폼 ============================================================
simulator_tab, dashboard_tab = st.tabs(
    [
        "예약 취소 위험 시뮬레이터",
        "모델 인사이트 대시보드"
    ]
)
with simulator_tab:
    input_col, result_col = st.columns(
        [1.8, 1],
        gap="large"
    )

    with input_col:
        with st.form("prediction_form"):

            st.subheader("예약 기본 정보")

            col1, col2 = st.columns(2)

            with col1:
                hotel = st.selectbox(
                    "호텔 유형",
                    category_options["hotel"]
                )

                meal = st.selectbox(
                    "식사 유형",
                    category_options["meal"]
                )

                reserved_room_type = st.selectbox(
                    "예약 객실 유형",
                    category_options["reserved_room_type"]
                )

                customer_type = st.selectbox(
                    "고객 유형",
                    category_options["customer_type"]
                )


            with col2:
                arrival_date_month = st.selectbox(
                    "도착 월",
                    category_options["arrival_date_month"]
                )

                market_segment = st.selectbox(
                    "예약 경로",
                    category_options["market_segment"]
                )

                country_group = st.selectbox(
                    "국가 그룹",
                    category_options["country_group"]
                )
                
                
            with st.container(border=True):
                st.subheader("예약 상세 정보")
                
                col1, col2 = st.columns(2)

                with col1:
                    lead_time = st.number_input(
                        "예약 후 체크인까지 남은 일수",
                        min_value=0,
                        value=30
                    )

                    total_nights = st.number_input(
                        "총 숙박일",
                        min_value=1,
                        value=2
                    )

                    required_car_parking_spaces = st.number_input(
                        "필요 주차 공간 수",
                        min_value=0,
                        value=0
                    )

                    total_of_special_requests = st.number_input(
                        "특별 요청 수",
                        min_value=0,
                        value=0
                    )                    

                with col2:
                    adr = st.number_input(
                        "ADR (평균 일일 객실 요금)",
                        min_value=0.0,
                        value=100.0
                    )

                    previous_bookings_not_canceled = st.number_input(
                        "이전 정상 예약 횟수",
                        min_value=0,
                        value=0
                    )
                    
                    previous_cancellations = st.number_input(
                        "이전 예약 취소 횟수",
                        min_value=0,
                        value=0
                    )                    

                
            with st.container(border=True):
                st.subheader("투숙객 정보")

                col1, col2, col3 = st.columns(3)

                with col1:
                    adults = st.number_input(
                        "성인",
                        min_value=1,
                        value=2
                    )

                with col2:
                    children = st.number_input(
                        "어린이",
                        min_value=0,
                        value=0
                    )

                with col3:
                    babies = st.number_input(
                        "유아",
                        min_value=0,
                        value=0
                    )

                total_guests = adults + children + babies
                has_children = int((children + babies) > 0)
            
            with st.container(border=True):
                st.subheader("기타 예약 정보")

                option_col1, option_col2, option_col3, option_col4 = st.columns(4)

                with option_col1:
                    is_repeated_guest = int(st.checkbox("재방문 고객"))

                with option_col2:
                    has_agent = int(st.checkbox("여행사/에이전트 예약"))

                with option_col3:
                    has_company = int(st.checkbox("회사 예약"))

                with option_col4:
                    has_waiting_list = int(st.checkbox("대기 목록 존재"))

            submitted = st.form_submit_button(
                "예약 취소 위험 분석",
                type="primary",
                use_container_width=True
            )
            
            if submitted:

                input_data = pd.DataFrame([{
                    "hotel": hotel,
                    "lead_time": lead_time,
                    "arrival_date_month": arrival_date_month,
                    "meal": meal,
                    "market_segment": market_segment,
                    "is_repeated_guest": is_repeated_guest,
                    "previous_cancellations": previous_cancellations,
                    "previous_bookings_not_canceled": previous_bookings_not_canceled,
                    "reserved_room_type": reserved_room_type,
                    "customer_type": customer_type,
                    "adr": adr,
                    "required_car_parking_spaces": required_car_parking_spaces,
                    "total_of_special_requests": total_of_special_requests,
                    "total_guests": total_guests,
                    "has_children": has_children,
                    "has_agent": has_agent,
                    "has_company": has_company,
                    "total_nights": total_nights,
                    "has_waiting_list": has_waiting_list,
                    "country_group": country_group
                }])
                
                
                cancel_probability = model.predict_proba(input_data)[0, 1]
                prediction = model.predict(input_data)[0]

                booking_amount = adr * total_nights

                expected_loss = (
                    cancel_probability * booking_amount
                )

                q50 = risk_thresholds["q50"]
                q75 = risk_thresholds["q75"]
                
                if expected_loss < q50:
                    risk_score = (expected_loss / q50) * 50
                    risk_level = "LOW"

                elif expected_loss < q75:
                    risk_score = 50 + ((expected_loss - q50) / (q75 - q50)) * 25
                    risk_level = "MEDIUM"

                else:
                    risk_score = 75 + ((expected_loss - q75) / q75) * 25
                    risk_level = "HIGH"
                    
                    
        with result_col:
            with st.container(border=True):
                st.subheader("예측 결과")

                if submitted:
                    st.metric(
                        label="예약 취소 확률",
                        value=f"{cancel_probability:.1%}"
                    )

                    st.progress(
                        min(cancel_probability, 1.0)
                    )

                    st.divider()

                    score_col1, score_col2 = st.columns(2)

                    with score_col1:
                        st.markdown('<div class="result-label">손실 위험 점수</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-value">{risk_score:.0f} / 100</div>', unsafe_allow_html=True)

                    with score_col2:
                        st.markdown('<div class="result-label">손실 위험 등급</div>', unsafe_allow_html=True)

                        if risk_level == "LOW":
                            st.markdown(
                                '<div class="risk-badge low">LOW</div>',
                                unsafe_allow_html=True
                            )

                        elif risk_level == "MEDIUM":
                            st.markdown(
                                '<div class="risk-badge medium">MEDIUM</div>',
                                unsafe_allow_html=True
                            )

                        else:
                            st.markdown(
                                '<div class="risk-badge high">HIGH</div>',
                                unsafe_allow_html=True
                            )

                    st.progress(
                        min(risk_score / 100, 1.0)
                    )

                    st.divider()

                    amount_col1, amount_col2 = st.columns(2)

                    with amount_col1:
                        st.metric(
                            "예약 금액 지표",
                            f"{booking_amount:,.2f}"
                        )

                    with amount_col2:
                        st.metric(
                            "예상 손실 지표",
                            f"{expected_loss:,.2f}"
                        )
                        
                    if risk_level == "LOW":
                        st.success(
                            "현재 예약은 상대적으로 낮은 손실 위험군으로 분류되었습니다."
                        )

                    elif risk_level == "MEDIUM":
                        st.warning(
                            "현재 예약은 중간 수준의 손실 위험군으로 분류되었습니다."
                        )

                    else:
                        st.error(
                            "현재 예약은 높은 손실 위험군으로 분류되었습니다."
                        )           
                else:
                    with st.container(border=True):
                        st.info(
                            "왼쪽의 예약 정보를 입력한 후 "
                            "'취소 위험 분석' 버튼을 눌러주세요."
                        )

with dashboard_tab:

    st.subheader("모델 인사이트 대시보드")

    st.caption(
        "모델 성능, 주요 예측 변수와 손실 위험등급의 검증 결과를 확인할 수 있습니다."
    )

    # 핵심 KPI

    final_result = rf_results[
        rf_results["Model"] == "Random Forest - Moderate Final"
    ].iloc[0]

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        with st.container(border=True):
            st.metric(
                "최종 모델 ROC-AUC",
                f"{final_result['Test ROC-AUC']:.4f}"
            )
            st.caption("취소 위험 구분 성능")

    with kpi2:
        with st.container(border=True):
            st.metric(
                "예측 정확도",
                f"{final_result['Test Accuracy']:.2%}"
            )
            st.caption("전체 예약 분류 정확도")

    with kpi3:
        with st.container(border=True):
            st.metric(
                "HIGH 위험군 실제 취소율",
                "74.5%"
            )
            st.caption("검증 데이터 기준")

    st.write("")

    # 모델 성능 + Permutation Importance
    performance_col, importance_col = st.columns(
        [1.15, 1],
        gap="large"
    )

    # 모델 성능
    with performance_col:
        with st.container(border=True):

            st.markdown("#### 모델 성능 비교")
            st.caption(
                "Random Forest 후보 모델의 테스트 성능을 비교합니다."
            )

            model_chart = rf_results[
                [
                    "Model",
                    "Test ROC-AUC",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ]
            ].copy()

            # 화면용 모델명 단축
            model_chart["Model"] = model_chart["Model"].replace({
                "Random Forest - Baseline": "Baseline",
                "Random Forest - Broad RandomizedSearchCV": "Broad Search",
                "Random Forest - Regularized Search": "Regularized",
                "Random Forest - Moderate Final": "Moderate Final"
            })

            metrics = [
                "Test ROC-AUC",
                "Precision",
                "Recall",
                "F1 Score"
            ]

            x = range(len(model_chart))

            fig, ax = plt.subplots(figsize=(9, 5))

            width = 0.18

            for i, metric in enumerate(metrics):
                positions = [
                    value + (i - 1.5) * width
                    for value in x
                ]

                ax.bar(
                    positions,
                    model_chart[metric],
                    width=width,
                    label=metric
                )

            ax.set_xticks(list(x))
            ax.set_xticklabels(
                model_chart["Model"],
                rotation=0
            )

            ax.set_ylim(0, 1)

            ax.set_ylabel("Score")
            ax.set_xlabel("")
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.27),
                ncol=2
            )

            ax.grid(
                axis="y",
                alpha=0.2
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


    # Permutation Importance
    with importance_col:
        with st.container(border=True):

            st.markdown("#### 취소 예측 주요 변수")
            st.caption(
                "Permutation Importance 기준 상위 10개 변수"
            )

            top_importance = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(10)
                .sort_values(
                    "Importance",
                    ascending=True
                )
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.barh(
                top_importance["Feature"],
                top_importance["Importance"]
            )

            ax.set_xlabel("Permutation Importance")
            ax.set_ylabel("")

            ax.grid(
                axis="x",
                alpha=0.2
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

    st.write("")

    # 손실 위험등급 검증
    st.markdown("### 손실 위험등급 검증")
    st.caption(
        "예상 손실 지표를 기준으로 분류한 위험등급과 실제 취소 결과를 비교합니다."
    )

    risk_cancel_col, risk_loss_col = st.columns(
        2,
        gap="large"
    )

    # 실제 취소율
    with risk_cancel_col:
        with st.container(border=True):

            st.markdown("#### 위험등급별 실제 취소율")

            risk_cancel_df = pd.DataFrame({
                "위험등급": [
                    "LOW",
                    "MEDIUM",
                    "HIGH"
                ],
                "실제 취소율": [
                    10.3,
                    53.8,
                    74.5
                ]
            })

            fig, ax = plt.subplots(figsize=(7, 4))

            bars = ax.bar(
                risk_cancel_df["위험등급"],
                risk_cancel_df["실제 취소율"]
            )

            ax.set_ylim(0, 100)
            ax.set_ylabel("실제 취소율 (%)")

            for bar, value in zip(
                bars,
                risk_cancel_df["실제 취소율"]
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 2,
                    f"{value:.1f}%",
                    ha="center"
                )

            ax.grid(
                axis="y",
                alpha=0.2
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


    # 평균 예상 손실
    with risk_loss_col:

        with st.container(border=True):

            st.markdown("#### 위험등급별 평균 예상 손실")

            risk_loss_df = pd.DataFrame({
                "위험등급": [
                    "LOW",
                    "MEDIUM",
                    "HIGH"
                ],
                "평균 예상 손실": [
                    22.269,
                    124.273,
                    387.696
                ]
            })

            fig, ax = plt.subplots(figsize=(7, 4))

            bars = ax.bar(
                risk_loss_df["위험등급"],
                risk_loss_df["평균 예상 손실"]
            )

            ax.set_ylabel("평균 예상 손실 지표")

            for bar, value in zip(
                bars,
                risk_loss_df["평균 예상 손실"]
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 8,
                    f"{value:.1f}",
                    ha="center"
                )

            ax.grid(
                axis="y",
                alpha=0.2
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

    st.write("")

    # 모델 상세 성능
    with st.container(border=True):

        st.markdown("#### 모델 상세 성능")

        st.caption(
            "Accuracy, Precision, Recall, F1, ROC-AUC를 "
            "후보 모델별로 비교합니다."
        )

        st.dataframe(
            rf_results.round(4),
            use_container_width=True,
            hide_index=True
        )