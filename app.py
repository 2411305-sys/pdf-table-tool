from io import BytesIO
from pathlib import Path
import os
import re

import pandas as pd
import pytesseract
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps


TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = Path(r"C:\Program Files\Tesseract-OCR\tessdata")

if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)

if TESSDATA_DIR.exists():
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


st.set_page_config(page_title="PDF 표 추출기", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.35rem !important;
    }

    h2, h3 {
        letter-spacing: 0 !important;
    }

    [data-testid="stFileUploader"] {
        margin-top: 0.35rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 116px;
        border-radius: 8px;
        border: 1px solid rgba(120, 130, 150, 0.42);
    }

    [data-testid="stFileUploaderDropzone"] button {
        min-width: 98px;
        font-size: 0 !important;
        color: transparent !important;
    }

    [data-testid="stFileUploaderDropzone"] button * {
        font-size: 0 !important;
        color: transparent !important;
    }

    [data-testid="stFileUploaderDropzone"] button::after {
        content: "파일 선택";
        color: #ffffff;
        display: inline-block;
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.2;
    }

    div[data-testid="stDownloadButton"] button,
    div[data-testid="stButton"] button {
        border-radius: 8px;
        min-height: 2.7rem;
        font-weight: 650;
    }

    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PDF/이미지 표 추출기")
st.caption("PDF나 스캔 이미지를 업로드하면 OCR 결과와 표 후보를 엑셀로 저장합니다.")

st.divider()

st.subheader("문서 업로드")
uploaded_file = st.file_uploader(
    "PDF 또는 이미지 파일",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    label_visibility="collapsed",
)

st.subheader("추출 설정")
left, middle, right = st.columns([1, 1, 1.15])

with left:
    ocr_lang = st.selectbox("OCR 언어", ["kor+eng", "kor", "eng"], index=0)

with middle:
    dpi = st.selectbox("PDF 변환 품질", [150, 200, 250, 300], index=2)

with right:
    min_columns = st.slider("표로 볼 최소 열 수", 2, 8, 3)


def load_images(file_name: str, file_bytes: bytes, pdf_dpi: int):
    extension = file_name.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        return convert_from_bytes(file_bytes, dpi=pdf_dpi)

    image = Image.open(BytesIO(file_bytes))
    return [image.convert("RGB")]


def prepare_image_for_ocr(image):
    prepared = ImageOps.grayscale(image)
    prepared = ImageOps.autocontrast(prepared)

    if prepared.width < 1800:
        ratio = 1800 / prepared.width
        prepared = prepared.resize(
            (1800, int(prepared.height * ratio)),
            Image.Resampling.LANCZOS,
        )

    prepared = prepared.filter(ImageFilter.SHARPEN)
    return prepared


def clean_ocr_words(dataframe):
    if dataframe.empty or "text" not in dataframe.columns:
        return pd.DataFrame()

    words = dataframe.copy()
    words["text"] = words["text"].fillna("").astype(str).str.strip()
    words["conf"] = pd.to_numeric(words["conf"], errors="coerce").fillna(-1)
    words = words[(words["text"] != "") & (words["text"].str.lower() != "nan")]
    words = words[words["conf"] >= 0]
    return words


def ocr_image(image, lang: str):
    config = "--psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(image, lang=lang, config=config)


def ocr_image_data(image, lang: str):
    config = "--psm 6 -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DATAFRAME,
    )
    return clean_ocr_words(data)


def parse_table_rows_from_text(text: str, min_col_count: int):
    rows = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        columns = [cell.strip() for cell in re.split(r"\s+|\t+", line) if cell.strip()]

        if len(columns) >= min_col_count:
            rows.append(columns)

    if not rows:
        return pd.DataFrame()

    max_columns = max(len(row) for row in rows)
    padded_rows = [row + [""] * (max_columns - len(row)) for row in rows]
    return pd.DataFrame(padded_rows)


def split_line_words_into_cells(line_words):
    line_words = line_words.sort_values("left")

    if len(line_words) <= 1:
        return line_words["text"].tolist()

    widths = line_words["width"].clip(lower=1)
    median_width = float(widths.median()) if not widths.empty else 20.0
    gap_threshold = max(22.0, median_width * 0.9)

    cells = []
    current_cell = []
    previous_right = None

    for _, word in line_words.iterrows():
        left = float(word["left"])
        right = left + float(word["width"])
        text = str(word["text"]).strip()

        if previous_right is not None and left - previous_right > gap_threshold:
            if current_cell:
                cells.append(" ".join(current_cell))
            current_cell = [text]
        else:
            current_cell.append(text)

        previous_right = right

    if current_cell:
        cells.append(" ".join(current_cell))

    return cells


def make_text_dataframe(page_results):
    rows = []

    for page_number, text, _ in page_results:
        for line_number, line in enumerate(text.splitlines(), start=1):
            clean_line = line.strip()

            if clean_line:
                rows.append(
                    {
                        "page": page_number,
                        "line": line_number,
                        "text": clean_line,
                    }
                )

    return pd.DataFrame(rows)


def make_coordinate_table_dataframe(page_results, min_col_count: int):
    rows = []

    for page_number, _, words in page_results:
        if words.empty:
            continue

        grouped = words.groupby(["block_num", "par_num", "line_num"], sort=True)

        for _, line_words in grouped:
            cells = split_line_words_into_cells(line_words)

            if len(cells) >= min_col_count:
                rows.append([f"p{page_number}"] + cells)

    if not rows:
        return pd.DataFrame()

    max_columns = max(len(row) for row in rows)
    padded_rows = [row + [""] * (max_columns - len(row)) for row in rows]
    return pd.DataFrame(padded_rows)


def make_excel(text_df, table_df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        text_df.to_excel(writer, sheet_name="OCR_Text", index=False)

        if table_df.empty:
            pd.DataFrame({"message": ["No table-like rows found"]}).to_excel(
                writer,
                sheet_name="Parsed_Table",
                index=False,
            )
        else:
            table_df.to_excel(writer, sheet_name="Parsed_Table", index=False, header=False)

    output.seek(0)
    return output.getvalue()


if uploaded_file is not None:
    st.success(f"업로드 완료: {uploaded_file.name}")

    if st.button("표 추출하기", type="primary", use_container_width=True):
        try:
            with st.spinner("OCR로 문서를 읽는 중입니다..."):
                images = load_images(uploaded_file.name, uploaded_file.getvalue(), dpi)
                page_results = []

                for page_number, image in enumerate(images, start=1):
                    prepared_image = prepare_image_for_ocr(image)
                    text = ocr_image(prepared_image, ocr_lang)
                    words = ocr_image_data(prepared_image, ocr_lang)
                    page_results.append((page_number, text, words))

                full_text = "\n".join(text for _, text, _ in page_results)
                text_df = make_text_dataframe(page_results)
                table_df = make_coordinate_table_dataframe(page_results, min_columns)

                if table_df.empty:
                    table_df = parse_table_rows_from_text(full_text, min_columns)

                excel_bytes = make_excel(text_df, table_df)

            st.success("OCR 처리가 끝났습니다.")

            result_left, result_right = st.columns([1.1, 1])

            with result_left:
                st.subheader("표 후보")
                if table_df.empty:
                    st.info("표처럼 나뉜 줄을 찾지 못했습니다. OCR_Text 시트를 확인해 주세요.")
                else:
                    st.dataframe(table_df, hide_index=True, use_container_width=True)

            with result_right:
                st.subheader("전체 OCR 텍스트")
                st.dataframe(text_df, hide_index=True, use_container_width=True)

            if table_df.empty:
                file_name = "ocr_text.xlsx"
            else:
                file_name = "extracted_tables.xlsx"

            st.download_button(
                "엑셀 다운로드",
                data=excel_bytes,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        except Exception as error:
            st.error("처리 중 오류가 발생했습니다.")
            st.exception(error)
else:
    st.info("PDF, PNG, JPG, TIF, BMP 파일을 업로드할 수 있습니다.")
