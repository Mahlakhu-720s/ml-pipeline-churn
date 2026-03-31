"""
Generate synthetic clinical discharge summary PDFs for the Healthcare RAG demo.
All patient data is entirely fabricated and marked [SYNTHETIC].
"""
import json
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT


PATIENT_PROFILES = [
    {
        "doc_id": "patient_001_diabetes_ckd",
        "name": "[SYNTHETIC] James Okafor",
        "mrn": "SYN-00001",
        "dob": "1958-04-22",
        "admission": "2024-01-10",
        "discharge": "2024-01-17",
        "physician": "[SYNTHETIC] Dr. Sarah Miller",
        "chief_complaint": "Uncontrolled hyperglycaemia and worsening renal function.",
        "hpi": (
            "Mr. Okafor is a 65-year-old male with a 12-year history of Type 2 Diabetes Mellitus "
            "and Stage 3 Chronic Kidney Disease who presented with blood glucose readings consistently "
            "above 18 mmol/L for 2 weeks. He reported polyuria, polydipsia, and fatigue. "
            "No chest pain or dyspnoea. He had missed several clinic appointments."
        ),
        "pmh": "Type 2 Diabetes Mellitus, Chronic Kidney Disease Stage 3, Hypertension, Dyslipidaemia.",
        "meds_on_admission": "Metformin 1000mg BD, Amlodipine 5mg daily, Atorvastatin 40mg nocte.",
        "exam": (
            "BP 158/94 mmHg, HR 82 bpm, Temp 36.8°C, SpO2 97% on room air. "
            "Alert and oriented. Mild ankle oedema bilaterally. No focal neurological deficit."
        ),
        "labs": (
            "HbA1c 9.8%, Fasting glucose 17.2 mmol/L, Creatinine 168 umol/L, eGFR 38 mL/min/1.73m2, "
            "Potassium 5.1 mmol/L, Sodium 137 mmol/L, Urea 12.4 mmol/L. "
            "Urine albumin-creatinine ratio 85 mg/mmol."
        ),
        "hospital_course": (
            "Metformin was withheld given CKD Stage 3 and eGFR < 45. Insulin therapy was initiated: "
            "Insulin Glargine 20 units nocte with Insulin Aspart sliding scale before meals. "
            "Renal dietitian review completed; low-potassium, low-phosphate diet advised. "
            "Lisinopril 10mg daily added for renoprotection and BP control. "
            "Glucose levels stabilised to 6.8-9.2 mmol/L by day 4. "
            "Nephrology consultation arranged for outpatient follow-up."
        ),
        "discharge_diagnosis": (
            "1. Uncontrolled Type 2 Diabetes Mellitus (HbA1c 9.8%).\n"
            "2. Chronic Kidney Disease Stage 3b (eGFR 38).\n"
            "3. Hypertension, improved."
        ),
        "discharge_meds": (
            "Insulin Glargine 20 units subcutaneous nocte, "
            "Insulin Aspart 4-8 units subcutaneous before meals (sliding scale), "
            "Lisinopril 10mg daily, Amlodipine 5mg daily, Atorvastatin 40mg nocte. "
            "Metformin DISCONTINUED."
        ),
        "followup": (
            "Endocrinology clinic in 4 weeks. Nephrology OPD in 6 weeks. "
            "Repeat HbA1c, renal function, and potassium in 4 weeks. "
            "Diabetic educator referral placed."
        ),
        "condition": "Stable, improved.",
    },
    {
        "doc_id": "patient_002_acute_mi",
        "name": "[SYNTHETIC] Priya Naidoo",
        "mrn": "SYN-00002",
        "dob": "1962-09-05",
        "admission": "2024-02-03",
        "discharge": "2024-02-08",
        "physician": "[SYNTHETIC] Dr. Themba Dlamini",
        "chief_complaint": "Acute chest pain radiating to left arm, onset 2 hours prior to arrival.",
        "hpi": (
            "Ms. Naidoo is a 61-year-old female with hypertension and smoking history (30 pack-years, "
            "quit 5 years ago) who presented with sudden onset severe retrosternal chest pain 8/10, "
            "radiating to the left arm and jaw, associated with diaphoresis and nausea. "
            "ECG on arrival showed ST-elevation in leads II, III, aVF."
        ),
        "pmh": "Hypertension, Hypercholesterolaemia, Ex-smoker.",
        "meds_on_admission": "Amlodipine 10mg daily, Rosuvastatin 20mg nocte.",
        "exam": (
            "BP 148/90 mmHg, HR 96 bpm, Temp 36.6°C, SpO2 96% on room air. "
            "Diaphoretic, in pain. JVP not elevated. S1 S2 heard, no murmurs. "
            "Basal crackles bilaterally."
        ),
        "labs": (
            "Troponin I peak 24.6 ng/mL (normal < 0.04), CK-MB 186 U/L, "
            "Total cholesterol 6.2 mmol/L, LDL 4.1 mmol/L, "
            "Creatinine 88 umol/L, eGFR > 60. INR 1.1."
        ),
        "hospital_course": (
            "Activated STEMI protocol. Dual antiplatelet therapy initiated: Aspirin 300mg loading then "
            "100mg daily, Ticagrelor 180mg loading then 90mg BD. Heparin infusion commenced. "
            "Emergency primary PCI performed: single drug-eluting stent to right coronary artery (RCA). "
            "TIMI 3 flow achieved post-intervention. "
            "Echo post-PCI: EF 45%, inferior wall hypokinesis. "
            "Commenced Ramipril 2.5mg daily, Bisoprolol 2.5mg daily, Atorvastatin 80mg nocte. "
            "Cardiac rehabilitation referral placed."
        ),
        "discharge_diagnosis": (
            "1. Inferior ST-Elevation Myocardial Infarction (STEMI).\n"
            "2. Single-vessel CAD — RCA, post-PCI with DES.\n"
            "3. Hypertension."
        ),
        "discharge_meds": (
            "Aspirin 100mg daily (lifelong), Ticagrelor 90mg BD (12 months minimum), "
            "Ramipril 2.5mg daily, Bisoprolol 2.5mg daily, Atorvastatin 80mg nocte, "
            "Amlodipine 10mg daily. GTN spray PRN."
        ),
        "followup": (
            "Cardiology OPD in 2 weeks. Cardiac rehabilitation programme referral. "
            "Repeat Echo in 6 weeks to reassess EF. "
            "Do NOT stop antiplatelet therapy without cardiology advice."
        ),
        "condition": "Stable, haemodynamically compensated.",
    },
    {
        "doc_id": "patient_003_chf_exacerbation",
        "name": "[SYNTHETIC] Robert van der Berg",
        "mrn": "SYN-00003",
        "dob": "1950-12-30",
        "admission": "2024-03-15",
        "discharge": "2024-03-22",
        "physician": "[SYNTHETIC] Dr. Ayanda Khumalo",
        "chief_complaint": "Progressive dyspnoea on exertion and bilateral leg swelling for 1 week.",
        "hpi": (
            "Mr. van der Berg is a 73-year-old male with known ischaemic cardiomyopathy (EF 30%) who "
            "presented with worsening shortness of breath on minimal exertion, orthopnoea (3 pillows), "
            "and bilateral ankle oedema increasing over 7 days. "
            "He admits non-compliance with fluid restriction and increased dietary salt intake."
        ),
        "pmh": (
            "Ischaemic Cardiomyopathy (EF 30%), AF on anticoagulation, "
            "CKD Stage 2, Hypertension, previous CABG 2018."
        ),
        "meds_on_admission": (
            "Furosemide 80mg daily, Carvedilol 25mg BD, Sacubitril/Valsartan 97/103mg BD, "
            "Spironolactone 25mg daily, Apixaban 5mg BD, Atorvastatin 40mg nocte."
        ),
        "exam": (
            "BP 132/86 mmHg, HR 88 bpm irregular, Temp 36.9°C, SpO2 91% on room air (94% on 2L O2). "
            "JVP elevated 5 cm above sternal angle. Bibasal crepitations. "
            "Bilateral pitting oedema to mid-shin. S3 gallop present."
        ),
        "labs": (
            "NT-proBNP 8420 pg/mL, Creatinine 124 umol/L, eGFR 52, "
            "Potassium 4.2 mmol/L, Sodium 132 mmol/L (mild hyponatraemia). "
            "CXR: cardiomegaly, bilateral pleural effusions, pulmonary vascular congestion."
        ),
        "hospital_course": (
            "IV Furosemide 120mg BD for 48 hours achieving negative fluid balance of 3.2L. "
            "Oxygen therapy maintained SpO2 > 94%. Daily weights and fluid balance charted. "
            "Renal function monitored closely: creatinine peaked at 138 umol/L then improved. "
            "Furosemide converted to oral 120mg daily on day 4. "
            "Fluid restriction 1.5L/day and low-salt diet reinforced by dietitian. "
            "Echo: EF 28%, stable. AF rate well controlled."
        ),
        "discharge_diagnosis": (
            "1. Acute decompensated heart failure (EF 28%).\n"
            "2. Atrial fibrillation, rate-controlled.\n"
            "3. CKD Stage 2, stable."
        ),
        "discharge_meds": (
            "Furosemide 120mg daily (increased from 80mg), Carvedilol 25mg BD, "
            "Sacubitril/Valsartan 97/103mg BD, Spironolactone 25mg daily, "
            "Apixaban 5mg BD, Atorvastatin 40mg nocte. "
            "Fluid restriction: 1.5L per day. Daily weight monitoring."
        ),
        "followup": (
            "Heart failure clinic in 1 week. Repeat renal function and electrolytes in 5 days. "
            "Daily weights — return to ED if weight increases > 2kg in 2 days."
        ),
        "condition": "Improved, euvolaemic at discharge.",
    },
    {
        "doc_id": "patient_004_pneumonia",
        "name": "[SYNTHETIC] Lindiwe Zulu",
        "mrn": "SYN-00004",
        "dob": "1985-07-18",
        "admission": "2024-04-02",
        "discharge": "2024-04-07",
        "physician": "[SYNTHETIC] Dr. Marcus Pretorius",
        "chief_complaint": "Productive cough, fever, and right-sided pleuritic chest pain for 4 days.",
        "hpi": (
            "Ms. Zulu is a 38-year-old female, non-smoker, no significant past medical history, "
            "who presented with 4 days of productive cough with green sputum, fever to 39.2°C, "
            "and right-sided pleuritic chest pain. She denied haemoptysis or recent travel."
        ),
        "pmh": "Nil significant. No regular medications. NKDA.",
        "meds_on_admission": "Nil.",
        "exam": (
            "BP 118/72 mmHg, HR 104 bpm, Temp 38.9°C, RR 22/min, SpO2 93% on room air. "
            "Reduced air entry and dullness to percussion right base. "
            "Bronchial breathing with crackles right lower zone."
        ),
        "labs": (
            "WBC 18.4 x10^9/L (neutrophilia), CRP 214 mg/L, Procalcitonin 2.8 ng/mL, "
            "Creatinine 72 umol/L. Blood cultures x2 sent. "
            "Sputum culture: Streptococcus pneumoniae (penicillin-sensitive). "
            "CXR: right lower lobe consolidation."
        ),
        "hospital_course": (
            "Commenced IV Amoxicillin-Clavulanate 1.2g TDS and Azithromycin 500mg daily. "
            "Supplemental oxygen to maintain SpO2 > 95%. IV fluids for rehydration. "
            "Sputum culture confirmed Strep pneumoniae, penicillin-sensitive. "
            "Narrowed to IV Amoxicillin 1g TDS day 2. Stepped down to oral Amoxicillin 500mg TDS day 4. "
            "Fever resolved day 3. CRP trending down to 68 mg/L at discharge. "
            "CURB-65 score was 1 on admission."
        ),
        "discharge_diagnosis": (
            "1. Community-acquired pneumonia (CAP), right lower lobe.\n"
            "2. Streptococcus pneumoniae bacteraemia (pending final blood culture result)."
        ),
        "discharge_meds": (
            "Amoxicillin 500mg TDS for 5 further days (completing 7-day course total). "
            "Paracetamol 1g QID PRN for fever/pain."
        ),
        "followup": (
            "GP review in 1 week. Repeat CXR in 6 weeks to confirm resolution. "
            "Return if fever recurs, dyspnoea worsens, or haemoptysis develops. "
            "Pneumococcal and influenza vaccination recommended."
        ),
        "condition": "Improved, afebrile at discharge.",
    },
    {
        "doc_id": "patient_005_hip_fracture",
        "name": "[SYNTHETIC] Elizabeth Mokoena",
        "mrn": "SYN-00005",
        "dob": "1944-03-07",
        "admission": "2024-05-20",
        "discharge": "2024-06-03",
        "physician": "[SYNTHETIC] Dr. Fatima Ismail",
        "chief_complaint": "Left hip pain and inability to weight-bear following a fall.",
        "hpi": (
            "Ms. Mokoena is an 80-year-old female who fell from standing height at home, "
            "landing on her left side. She was unable to weight-bear and complained of severe "
            "left hip pain. No loss of consciousness. On arrival left leg was shortened and "
            "externally rotated."
        ),
        "pmh": "Osteoporosis, Hypertension, Hypothyroidism, Type 2 Diabetes Mellitus.",
        "meds_on_admission": (
            "Alendronate 70mg weekly, Calcium/Vitamin D supplement daily, "
            "Perindopril 8mg daily, Levothyroxine 75mcg daily, Metformin 500mg BD."
        ),
        "exam": (
            "BP 144/88 mmHg, HR 78 bpm, Temp 36.7°C, SpO2 97% on room air. "
            "Left lower limb shortened and externally rotated. "
            "Marked tenderness over left hip. Neurovascular status intact distally."
        ),
        "labs": (
            "Hb 10.8 g/dL (normocytic anaemia), WBC 9.2 x10^9/L, Platelets 234 x10^9/L. "
            "Creatinine 92 umol/L, eGFR 58. HbA1c 7.4%. "
            "Calcium 2.28 mmol/L, Vitamin D 42 nmol/L (insufficient). "
            "X-ray pelvis: left intracapsular femoral neck fracture (Garden III)."
        ),
        "hospital_course": (
            "Orthopaedic surgery performed day 1: total hip replacement (THR) under spinal anaesthesia. "
            "Cefazolin 2g IV prophylaxis given. Estimated blood loss 420 mL. "
            "Postoperative Hb 9.1 g/dL — transfusion threshold not met, IV iron commenced. "
            "Enoxaparin 40mg daily for VTE prophylaxis (6 weeks post-discharge). "
            "Physiotherapy commenced day 1 post-op; mobilising with frame by day 3. "
            "Occupational therapy: home assessment arranged. "
            "Endocrinology review: Metformin held peri-operatively, restarted day 3 post-op. "
            "Vitamin D supplementation increased to Cholecalciferol 50,000 IU weekly x 8 weeks. "
            "Falls prevention plan documented."
        ),
        "discharge_diagnosis": (
            "1. Left intracapsular femoral neck fracture (Garden III), post-THR.\n"
            "2. Osteoporosis.\n"
            "3. Iron deficiency anaemia, post-operative.\n"
            "4. Type 2 Diabetes Mellitus, well-controlled."
        ),
        "discharge_meds": (
            "Enoxaparin 40mg subcutaneous daily (6 weeks), "
            "Alendronate 70mg weekly (RESUME after 6 weeks post-op), "
            "Calcium/Vitamin D supplement daily, "
            "Cholecalciferol 50,000 IU weekly x 8 weeks, "
            "Ferrous sulphate 200mg BD, "
            "Perindopril 8mg daily, Levothyroxine 75mcg daily, Metformin 500mg BD. "
            "Paracetamol 1g QID regular for 2 weeks, then PRN."
        ),
        "followup": (
            "Orthopaedic OPD in 2 weeks (wound check and X-ray). "
            "Physiotherapy 3 x per week for 6 weeks. "
            "DEXA scan in 3 months to reassess bone density. "
            "Repeat FBC and iron studies in 4 weeks. "
            "Falls and fracture liaison service referral placed."
        ),
        "condition": "Stable, mobilising with frame at discharge.",
    },
]


EVAL_QA_PAIRS = [
    # Patient 001 — Diabetes + CKD
    {
        "question": "What were the discharge medications for the diabetic patient with CKD?",
        "ground_truth": (
            "Insulin Glargine 20 units subcutaneous nocte, Insulin Aspart 4-8 units subcutaneous "
            "before meals (sliding scale), Lisinopril 10mg daily, Amlodipine 5mg daily, "
            "Atorvastatin 40mg nocte. Metformin was discontinued."
        ),
        "source_doc_id": "patient_001_diabetes_ckd",
    },
    {
        "question": "Why was Metformin stopped for the diabetic patient?",
        "ground_truth": (
            "Metformin was withheld and later discontinued due to the patient's Chronic Kidney Disease "
            "Stage 3 with eGFR below 45 mL/min/1.73m2."
        ),
        "source_doc_id": "patient_001_diabetes_ckd",
    },
    # Patient 002 — Acute MI
    {
        "question": "What procedure was performed for the STEMI patient?",
        "ground_truth": (
            "Emergency primary percutaneous coronary intervention (PCI) was performed with a single "
            "drug-eluting stent placed in the right coronary artery (RCA), achieving TIMI 3 flow."
        ),
        "source_doc_id": "patient_002_acute_mi",
    },
    {
        "question": "What antiplatelet therapy was prescribed after the heart attack?",
        "ground_truth": (
            "Aspirin 100mg daily (lifelong) and Ticagrelor 90mg twice daily for a minimum of 12 months."
        ),
        "source_doc_id": "patient_002_acute_mi",
    },
    # Patient 003 — CHF
    {
        "question": "What was the patient's ejection fraction and main heart failure diagnosis?",
        "ground_truth": (
            "The patient had an ejection fraction of 28-30% and was diagnosed with acute decompensated "
            "heart failure due to ischaemic cardiomyopathy."
        ),
        "source_doc_id": "patient_003_chf_exacerbation",
    },
    {
        "question": "What was the Furosemide dose changed to at discharge for the heart failure patient?",
        "ground_truth": "Furosemide was increased from 80mg daily to 120mg daily at discharge.",
        "source_doc_id": "patient_003_chf_exacerbation",
    },
    # Patient 004 — Pneumonia
    {
        "question": "What organism caused the pneumonia and how was it treated?",
        "ground_truth": (
            "Streptococcus pneumoniae (penicillin-sensitive) was identified. Treatment was narrowed to "
            "IV Amoxicillin 1g TDS after initial broad-spectrum cover, then stepped down to oral "
            "Amoxicillin 500mg TDS."
        ),
        "source_doc_id": "patient_004_pneumonia",
    },
    {
        "question": "What antibiotic was the pneumonia patient discharged with?",
        "ground_truth": "Amoxicillin 500mg three times daily to complete a 7-day course total.",
        "source_doc_id": "patient_004_pneumonia",
    },
    # Patient 005 — Hip Fracture
    {
        "question": "What surgery was performed for the hip fracture patient and what VTE prophylaxis was given?",
        "ground_truth": (
            "Total hip replacement (THR) was performed under spinal anaesthesia. "
            "Enoxaparin 40mg subcutaneous daily was prescribed for 6 weeks post-discharge for VTE prophylaxis."
        ),
        "source_doc_id": "patient_005_hip_fracture",
    },
    {
        "question": "Why was Alendronate withheld initially for the hip fracture patient?",
        "ground_truth": (
            "Alendronate was instructed to resume after 6 weeks post-operatively, "
            "as it is held in the immediate post-operative period following total hip replacement."
        ),
        "source_doc_id": "patient_005_hip_fracture",
    },
]


def build_discharge_summary(profile: dict) -> str:
    lines = [
        f"DISCHARGE SUMMARY",
        f"",
        f"Patient: {profile['name']}  |  MRN: {profile['mrn']}  |  DOB: {profile['dob']}",
        f"Admission Date: {profile['admission']}  |  Discharge Date: {profile['discharge']}",
        f"Attending Physician: {profile['physician']}",
        f"",
        f"CHIEF COMPLAINT:",
        f"  {profile['chief_complaint']}",
        f"",
        f"HISTORY OF PRESENT ILLNESS:",
        f"  {profile['hpi']}",
        f"",
        f"PAST MEDICAL HISTORY:",
        f"  {profile['pmh']}",
        f"",
        f"MEDICATIONS ON ADMISSION:",
        f"  {profile['meds_on_admission']}",
        f"",
        f"PHYSICAL EXAMINATION:",
        f"  {profile['exam']}",
        f"",
        f"DIAGNOSTIC RESULTS:",
        f"  {profile['labs']}",
        f"",
        f"HOSPITAL COURSE:",
        f"  {profile['hospital_course']}",
        f"",
        f"DISCHARGE DIAGNOSIS:",
        f"  {profile['discharge_diagnosis']}",
        f"",
        f"DISCHARGE MEDICATIONS:",
        f"  {profile['discharge_meds']}",
        f"",
        f"FOLLOW-UP INSTRUCTIONS:",
        f"  {profile['followup']}",
        f"",
        f"CONDITION AT DISCHARGE:",
        f"  {profile['condition']}",
    ]
    return "\n".join(lines)


def generate_pdf(text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=4
    )

    story = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
        elif stripped == "DISCHARGE SUMMARY":
            story.append(Paragraph(stripped, title_style))
        elif stripped.isupper() and stripped.endswith(":"):
            story.append(Paragraph(f"<b>{stripped}</b>", body_style))
        else:
            safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, body_style))

    doc.build(story)
    return output_path


def main(
    output_dir: Path = Path("rag_clinical_notes/data/sample_notes"),
    eval_path: Path = Path("rag_clinical_notes/data/eval_qa_pairs.json"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for profile in PATIENT_PROFILES:
        text = build_discharge_summary(profile)
        pdf_path = output_dir / f"{profile['doc_id']}.pdf"
        generate_pdf(text, pdf_path)
        print(f"  Created: {pdf_path}")

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(EVAL_QA_PAIRS, f, indent=2)
    print(f"  Created: {eval_path} ({len(EVAL_QA_PAIRS)} Q&A pairs)")
    print("Done.")


if __name__ == "__main__":
    main()
