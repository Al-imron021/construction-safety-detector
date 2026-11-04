
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import os

st.set_page_config(
    page_title="Construction Safety Detector",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Construction Safety Detector")
st.write("ระบบตรวจจับความปลอดภัยในพื้นที่ก่อสร้างด้วย YOLO26")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

if not os.path.exists("best.pt"):
    st.error("ไม่พบไฟล์ best.pt")
    st.stop()

model = load_model()

st.success("YOLO26 Model พร้อมใช้งาน")

uploaded_file = st.file_uploader(
    "เลือกรูปภาพที่ต้องการตรวจจับ",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "ระดับความมั่นใจ (Confidence)",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("ภาพต้นฉบับ")
    st.image(image, use_container_width=True)

    if st.button("🔍 เริ่มตรวจจับวัตถุ", type="primary"):

        with st.spinner("กำลังตรวจจับวัตถุ..."):

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

            result = results[0]

            # แสดงภาพที่มีกรอบตรวจจับ
            annotated = result.plot()
            annotated = annotated[:, :, ::-1]

            st.subheader("ผลการตรวจจับ")
            st.image(
                annotated,
                caption="YOLO26 Detection Result",
                use_container_width=True
            )

            # นับจำนวนวัตถุ
            if result.boxes is not None and len(result.boxes) > 0:

                class_ids = result.boxes.cls.int().tolist()
                confidences = result.boxes.conf.tolist()

                names = [
                    result.names[class_id]
                    for class_id in class_ids
                ]

                counts = Counter(names)

                st.subheader("สรุปผลการตรวจจับ")

                st.metric(
                    "จำนวนวัตถุทั้งหมด",
                    len(names)
                )

                for name, count in counts.items():
                    st.write(f"**{name}** : {count} รายการ")

                st.subheader("รายละเอียดแต่ละวัตถุ")

                for i, (name, conf) in enumerate(
                    zip(names, confidences), start=1
                ):
                    st.write(
                        f"{i}. {name} - ความมั่นใจ {conf*100:.2f}%"
                    )

            else:
                st.warning("ไม่พบวัตถุในภาพ")

st.divider()
st.caption("Developed with Python, Streamlit and YOLO26")
