# -*- coding: utf-8 -*-
"""성능평가 결과 → 논문용 docx 표 생성 (make_report.py 스타일 재사용)."""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from dataset_kpt import LABEL_NAMES

KOR_FONT = "맑은 고딕"


def set_kor_font(run, name=KOR_FONT):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)


def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    set_kor_font(run)
    if level == 1:
        run.font.size = Pt(16); run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    elif level == 2:
        run.font.size = Pt(13); run.font.color.rgb = RGBColor(0x2E, 0x54, 0x96)
    else:
        run.font.size = Pt(11.5); run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    return h


def add_para(doc, text, size=10.5, bold=False, italic=False, align=None,
             color=None, space_after=6):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    set_kor_font(run)
    run.font.size = Pt(size); run.font.bold = bold; run.font.italic = italic
    if color:
        run.font.color.rgb = color
    return p


def add_bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    set_kor_font(run)
    run.font.size = Pt(size)
    return p


def add_table(doc, headers, rows, widths=None, highlight_rows=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        shade_cell(hdr[i], "2E5496")
        para = hdr[i].paragraphs[0]; para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(h); set_kor_font(run)
        run.font.size = Pt(10); run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    highlight_rows = highlight_rows or set()
    for ri, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            if ri in highlight_rows:
                shade_cell(cells[i], "EAF0FA")
            para = cells[i].paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = para.add_run(str(val)); set_kor_font(run)
            run.font.size = Pt(9.5)
            if ri in highlight_rows:
                run.font.bold = True
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def _pct(x):
    return f"{x*100:.2f}"


def _lat(ms, tp):
    if ms is None:
        return "—"
    return f"{ms:.2f} ms ({tp:.1f}/s)"


def build_report(out_path, kpt, conv, device):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = KOR_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), KOR_FONT)

    # 제목
    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(6)
    r = title.add_run("이상행동 분류 모델 성능 평가")
    set_kor_font(r); r.font.size = Pt(18); r.font.bold = True
    r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("KeypointLSTMv2 vs. ConvLSTM(ResNet50+LSTM) 비교 평가")
    set_kor_font(r); r.font.size = Pt(11); r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    gpu_name = device if device == "cpu" else "CUDA GPU"
    add_para(doc,
             f"측정 환경: PyTorch {__import__('torch').__version__}, "
             f"{__import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'CPU'} / "
             "분류 지표는 혼동행렬 기반 실측값(sklearn 미사용, 직접 산출).",
             size=9, italic=True, color=RGBColor(0x60, 0x60, 0x60), space_after=10)

    # ── 1. 평가 개요 ──
    add_heading(doc, "1. 평가 개요", 1)
    add_para(doc,
             "본 평가는 8종 이상행동(정상·전도·파손·방화·흡연·유기·절도·폭행)을 분류하는 "
             "두 모델을 대상으로 한다. 두 모델은 입력 양식과 데이터 분할이 서로 달라 동일 "
             "테스트셋으로 직접 비교할 수 없으므로, 각 모델을 자신의 held-out 평가셋에서 "
             "동일한 지표 체계로 측정하였다. (ConvLSTM용 별도 test.csv는 원본 NIA 소스 "
             "경로를 가리켜 현재 이미지가 존재하지 않으므로, 실재하는 검증셋 val.csv로 "
             "평가하였다.)")
    add_table(doc,
              ["구분", "KeypointLSTMv2", "ConvLSTM (LSTM_NIA)"],
              [
                  ["모델 파일", "kpt_behavior.pth", "model_final.pth"],
                  ["아키텍처", kpt["arch"], conv["arch"]],
                  ["입력 양식", "키포인트 30프레임 (17관절×2)", "RGB 3프레임 (3×224×224)"],
                  ["파라미터 수", f"{kpt['n_params']:,}", f"{conv['n_params']:,}"],
                  ["학습 에포크", f"{kpt.get('epoch','-')} (best 배포)",
                   f"{conv.get('epoch','-')} (조기 체크포인트)"],
                  ["평가셋(등장 클래스)",
                   f"{kpt['eval_set']} ({kpt['n_samples']}개 / {kpt['n_present_classes']}종)",
                   f"{conv['eval_set']} ({conv['n_samples']}개 / {conv['n_present_classes']}종)"],
              ],
              widths=[1.5, 2.4, 2.4])

    # ── 2. 종합 성능평가표 ──
    add_heading(doc, "2. 성능 평가표", 1)
    add_para(doc, "■ 표 1. 분류 성능 및 추론 속도 종합 비교", bold=True, size=11,
             color=RGBColor(0x2E, 0x54, 0x96))
    rows = [
        ["Accuracy (%)", _pct(kpt["accuracy"]), _pct(conv["accuracy"])],
        ["Macro Precision (%)", _pct(kpt["macro"]["precision"]), _pct(conv["macro"]["precision"])],
        ["Macro Recall (%)", _pct(kpt["macro"]["recall"]), _pct(conv["macro"]["recall"])],
        ["Macro F1 (%)", _pct(kpt["macro"]["f1"]), _pct(conv["macro"]["f1"])],
        ["Weighted Precision (%)", _pct(kpt["weighted"]["precision"]), _pct(conv["weighted"]["precision"])],
        ["Weighted Recall (%)", _pct(kpt["weighted"]["recall"]), _pct(conv["weighted"]["recall"])],
        ["Weighted F1 (%)", _pct(kpt["weighted"]["f1"]), _pct(conv["weighted"]["f1"])],
        ["추론 지연 (GPU)", _lat(kpt["gpu_ms"], kpt["gpu_tp"]), _lat(conv["gpu_ms"], conv["gpu_tp"])],
        ["추론 지연 (CPU)", _lat(kpt["cpu_ms"], kpt["cpu_tp"]), _lat(conv["cpu_ms"], conv["cpu_tp"])],
    ]
    add_table(doc, ["지표", "KeypointLSTMv2", "ConvLSTM"], rows,
              widths=[2.2, 2.0, 2.0], highlight_rows={0, 3})
    add_para(doc,
             "※ Macro 평균은 표본이 존재하는 클래스에 대한 단순 평균, Weighted 평균은 "
             "클래스별 표본 수로 가중한 평균이다. 추론 지연은 배치=1 단일 추론의 순수 "
             "forward 시간(데이터 로딩·이미지 디코딩 제외)을 워밍업 후 다회 평균한 값이다.",
             size=9, italic=True, color=RGBColor(0x60, 0x60, 0x60), space_after=10)

    # ── 3. 클래스별 지표 ──
    add_heading(doc, "3. 클래스별 상세 지표", 1)
    for title_kr, m in (("표 2. KeypointLSTMv2 (kpt_behavior.pth)", kpt),
                        ("표 3. ConvLSTM (model_final.pth)", conv)):
        add_para(doc, f"■ {title_kr} — 평가셋 {m['eval_set']}", bold=True, size=11,
                 color=RGBColor(0x2E, 0x54, 0x96))
        pc = m["per_class"]
        crows = []
        for i, nm in enumerate(LABEL_NAMES):
            sup = int(pc["support"][i])
            if sup == 0:
                crows.append([f"{i} {nm}", "0", "—", "—", "—"])
            else:
                crows.append([f"{i} {nm}", str(sup),
                              _pct(pc["precision"][i]), _pct(pc["recall"][i]),
                              _pct(pc["f1"][i])])
        add_table(doc, ["클래스", "표본수", "Precision(%)", "Recall(%)", "F1(%)"],
                  crows, widths=[1.5, 1.0, 1.4, 1.4, 1.4])

    # ── 4. 분석 ──
    add_heading(doc, "4. 결과 분석", 1)
    kpt_normal_f1 = kpt["per_class"]["f1"][0] if kpt["per_class"]["support"][0] > 0 else 0.0
    add_bullet(doc,
               f"ConvLSTM(model_final.pth)의 다수 클래스 붕괴: 이 모델은 epoch "
               f"{conv.get('epoch','?')}의 조기 체크포인트로 학습이 충분히 진행되지 않아, "
               "입력과 무관하게 모든 표본을 '정상'으로만 예측하는 다수 클래스 붕괴"
               "(majority-class collapse) 상태이다. 평가셋(val.csv)이 정상 표본을 포함하지 "
               f"않는 이상행동 전용 셋이므로, '정상'만 출력하는 본 모델의 Accuracy는 "
               f"{_pct(conv['accuracy'])}%에 머물러 단 하나의 이상행동도 탐지하지 못한다. "
               "표 3에서 모든 클래스의 F1이 0%인 점이 이를 명확히 보여준다.")
    add_bullet(doc,
               f"정확도(Accuracy) 지표의 함정: Accuracy는 평가셋의 클래스 구성에 좌우되어 "
               "모델의 실제 분류 능력을 왜곡할 수 있다(위 ConvLSTM이 대표 사례). 클래스 "
               f"불균형에 영향받지 않는 Macro F1 기준으로는 KeypointLSTMv2 "
               f"{_pct(kpt['macro']['f1'])}% vs ConvLSTM {_pct(conv['macro']['f1'])}%이며, "
               "KeypointLSTMv2만이 정상 포함 8종을 실제로 구분한다(표 2). 이상행동 분류에서는 "
               "Accuracy 단독이 아닌 Macro F1·클래스별 지표 병기가 필수임을 보여준다.")
    add_bullet(doc,
               f"정상 데이터 보강 효과: '정상' 클래스를 매장이동·구매·반품 영상으로 보강해 "
               f"재학습한 결과, KeypointLSTMv2의 '정상' F1이 {_pct(kpt_normal_f1)}%로 "
               "확보되었다(표 2). 정상 표본이 부족해 '정상' 인식이 사실상 붕괴했던 이전 "
               "학습 대비, 정상/이상 구분의 기반이 마련된 것이다.")
    add_bullet(doc,
               f"모델 경량성: KeypointLSTMv2는 파라미터 {kpt['n_params']:,}개로 "
               f"ConvLSTM({conv['n_params']:,}개) 대비 약 {conv['n_params']//kpt['n_params']}배 "
               f"가벼워, CPU 추론 지연이 {kpt['cpu_ms']:.2f}ms 대 {conv['cpu_ms']:.1f}ms로 "
               "크게 빠르다(Raspberry Pi 5 CPU 실시간 구동에 유리). 시스템이 ConvLSTM에서 "
               "키포인트 기반 모델로 전환한 설계 결정의 근거가 된다.")
    add_bullet(doc,
               "한계: KeypointLSTMv2 역시 약지도(영상 전체 단일 라벨) 학습과 클래스 "
               f"불균형으로 절대 정확도는 제한적이다(train acc 대비 val acc 격차로 과적합 "
               "경향). 또한 평가셋이 모델별로 다르므로"
               f"(KeypointLSTMv2 {kpt['n_present_classes']}종 / ConvLSTM "
               f"{conv['n_present_classes']}종), 본 표는 동일 테스트셋 비교가 아닌 각 모델의 "
               "자체 held-out 평가임에 유의한다.")

    doc.save(out_path)
