import streamlit as st
from llama_cpp import Llama
import json

# --- 定数設定 ---
MODEL_PATH = "./Mistral-Nemo-Japanese-Instruct-2408-Q8_0.gguf"

# --- 問診項目テンプレート（変更なし） ---
QUESTIONS = [
    {"id": "q1", "text": "これからMRI検査を受けられますか？", "response_type": "yes_no", "end_if_no": True},
    {"id": "q_exam_type", "text": "これから受けられる検査は、造影剤を使用する「造影検査」ですか？それとも造影剤を使用しない「単純検査」ですか？\n\nもし不明な場合は「わからない」を選択してください。", "response_type": "exam_type"},
    {"id": "q_asthma", "text": "気管支喘息と診断されたことはありますか？", "response_type": "yes_no", "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_yes": "ご申告ありがとうございます。安全のため、当日に担当者が改めて確認させていただきます。"},
    {"id": "q_allergy", "text": "今までに薬や食べ物でアレルギー（発疹、かゆみ、気分不快など）を起こしたことはありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。差し支えなければ、何に対するアレルギーか教えていただけますか？", "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_detail": "重要な情報をありがとうございます。"},
    {"id": "q_kidney", "text": "以前に腎臓の病気を指摘されたことはありますか？", "response_type": "yes_no", "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_yes": "ご申告ありがとうございます。安全のため、当日に担当者が改めて確認させていただきます。"},
    {"id": "q_contrast_side_effect", "text": "以前にMRIやCTの造影検査で気分が悪くなったり、副作用が出たりしたことはありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。どのような症状が出たか具体的に教えていただけますか？", "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_detail": "詳細な情報をありがとうございます。こちらも安全のため、当日改めて確認させていただきます。"},
    {"id": "q_consent_form", "text": "「MRI造影検査 同意書兼問診票」に、ご自身で署名されましたか？", "response_type": "yes_no", "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_no": "承知いたしました。安全な検査のために同意書の署名は必須となります。当日、担当者が改めてご案内・確認いたしますのでご安心ください。"},
    {"id": "q_previous_mri", "text": "以前にMRI検査を受けたことはありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。その検査は、**今回と同じ部位ですか、それとも別の部位ですか？** また、もし分かれば**いつ頃**受けられたかも教えていただけますか？"},
    {"id": "q_pacemaker", "text": "心臓ペースメーカー、植え込み型除細動器（ICD）が体内にありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。どちらが体内にあるか、また、もし可能であればその製品名や手術日を教えていただけますか？", "follow_up_on_detail": "重要な情報をありがとうございます。安全を最優先するため、当日に担当者が改めて詳細を確認させていただきますのでご安心ください。"},
    {"id": "q_brain_clip", "text": "脳動脈瘤クリップやコイルが体内にありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。材質（チタン製など）や手術日について、もし分かれば教えてください。", "follow_up_on_detail": "詳細な情報をありがとうございます。こちらも安全のため、当日改めて確認させていただきます。"},
    {"id": "q_metal", "text": "人工関節、骨折プレート、ボルト、ワイヤーなどの金属が体内にありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。どの部位に、どのような金属が入っているか教えていただけますか？", "follow_up_on_detail": "ご回答ありがとうございます。"},
    {"id": "q_pregnancy", "text": "現在、妊娠されている可能性はありますか？", "response_type": "yes_no", "follow_up_on_yes": "ご申告ありがとうございます。妊娠中のMRI検査は慎重な判断が必要なため、当日担当者が改めて詳細をお伺いいたします。"},
    {"id": "q_claustrophobia", "text": "現在、狭い場所や閉鎖的な場所が苦手ですか？（閉所恐怖症）", "response_type": "yes_no", "follow_up_on_yes": "ご申告ありがとうございます。検査中はスタッフが常にマイクやカメラで様子を見ており、いつでも会話ができます。また、ご気分が悪くなった際には、緊急ブザーでお知らせいただけますのでご安心ください。"},
    {"id": "q_other_concerns", "text": "他に何か、ご心配なことやご質問はありますか？", "response_type": "dialogue"},
]

# --- AIモデルのロード（変更なし） ---
@st.cache_resource
def load_model():
    try:
        return Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, verbose=False, chat_format="mistral-instruct")
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}"); st.stop()
llm = load_model()

# --- LLMの役割を限定した関数（変更なし） ---
def classify_user_intent(user_input):
    prompt = f"""[INST] <<SYS>>
あなたはテキスト分類アシスタントです。患者の発言の意図を「質問・懸念あり(question)」か「特にない・終了(end)」に分類し、JSON形式で{{"intent": "..."}}とだけ出力してください。
<</SYS>>
患者の発言: {user_input}
[/INST]"""
    try:
        output = llm(prompt, max_tokens=32, temperature=0.0)
        response_text = output["choices"][0]["text"]
        start_index = response_text.find('{')
        end_index = response_text.rfind('}') + 1
        if start_index != -1 and end_index != -1:
            json_str = response_text[start_index:end_index]
            data = json.loads(json_str)
            return data.get("intent", "question")
        return "question"
    except Exception: return "question"

def answer_general_question(user_question):
    prompt = f"""[INST] <<SYS>>
あなたはMRI検査に関する一般的な質問に答える、親切なAIアシスタントです。患者の質問に、簡潔かつ正確に、1-2文で答えてください。
<</SYS>>
質問: {user_question}
[/INST]"""
    try:
        output = llm(prompt, max_tokens=256, temperature=0.7, stop=["</s>", "[INST]"])
        return output["choices"][0]["text"].strip()
    except Exception: return "申し訳ありません、現在回答を生成できません。"

# --- Streamlit App（変更なし） ---
st.title("MRI AI 問診 🏥")
st.caption("AIとルールベースのハイブリッドで、安定かつ柔軟な問診を実現します。")

# --- 初期化処理（変更なし） ---
if "stage" not in st.session_state:
    st.session_state.stage = "asking_main"; st.session_state.q_index = 0
    st.session_state.answers = {}; st.session_state.messages = []
    first_q = QUESTIONS[0]["text"]
    initial_message = f"こんにちは。MRI検査を安全に行うため、いくつか質問をさせていただきます。\n\n{first_q}"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# --- 質問遷移ロジック（変更なし） ---
def move_to_next_question():
    current_q_index = st.session_state.q_index
    st.session_state.q_index += 1
    while st.session_state.q_index < len(QUESTIONS):
        next_q_info = QUESTIONS[st.session_state.q_index]
        should_skip = False
        if "skip_if" in next_q_info:
            condition = next_q_info["skip_if"]
            if (condition["q_id"] in st.session_state.answers and
                st.session_state.answers[condition["q_id"]]["type"] == condition["answer_type"]):
                should_skip = True
        if should_skip:
            st.session_state.q_index += 1; continue
        else:
            # 最後の質問からの遷移時は「承知いたしました」などを言わないようにする
            if QUESTIONS[current_q_index]['response_type'] == 'dialogue':
                message = f"{next_q_info['text']}"
            else:
                message = f"承知いたしました。では次の質問です。\n\n{next_q_info['text']}"
            st.session_state.messages.append({"role": "assistant", "content": message})
            return
    st.session_state.stage = "finished"
    final_summary = "ご協力ありがとうございました。以下が問診の回答サマリーです。\n\n---\n\n"
    for q_info in QUESTIONS:
        q_id = q_info["id"]
        if q_id in st.session_state.answers:
            answer_data = st.session_state.answers[q_id]
            final_summary += f"**Q: {q_info['text']}**\nA: {answer_data.get('answer', '（回答なし）')}"
            if "detail" in answer_data: final_summary += f"\n> **詳細:** {answer_data['detail']}\n\n"
            elif "details" in answer_data:
                 final_summary += "\n> **追加の質問と懸念事項:**\n"
                 for item in answer_data.get('details', []): final_summary += f"> - {item}\n"
                 final_summary += "\n"
            else: final_summary += "\n\n"
    st.session_state.messages.append({"role": "assistant", "content": final_summary})

# --- ★★★ 回答処理ロジック（ここを修正） ★★★ ---
def process_answer(user_input, **kwargs):
    q_index = st.session_state.q_index
    current_q_info = QUESTIONS[q_index]
    q_id = current_q_info["id"]
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if st.session_state.stage == "asking_detail":
        st.session_state.answers[q_id]["detail"] = user_input
        if "follow_up_on_detail" in current_q_info: st.session_state.messages.append({"role": "assistant", "content": current_q_info["follow_up_on_detail"]})
        st.session_state.stage = "asking_main"; move_to_next_question()

    elif st.session_state.stage == "dialogue":
        # ルールベースのチェックを先に行う
        end_phrases = ["特にありません", "ありません", "ないです", "大丈夫です", "以上です", "ない"]
        normalized_input = user_input.strip().replace("。", "").replace("、", "")

        # 定型句に一致すれば、LLMを呼ばずに終了処理へ
        if normalized_input in end_phrases:
            st.session_state.answers[q_id] = {"answer": "特になし"}
            st.session_state.stage = "asking_main" # stageを戻す
            move_to_next_question() # これで最終サマリーが呼ばれる
            return # ここで処理を終了

        # 定型句でない場合のみ、LLMに意図分類を依頼
        with st.spinner("内容を確認しています..."): intent = classify_user_intent(user_input)
        if intent == "end":
            st.session_state.answers[q_id] = {"answer": user_input} # 念のためユーザーの最後の言葉も記録
            st.session_state.stage = "asking_main"; move_to_next_question()
        else: # intentが'question'だった場合
            if st.session_state.answers.get(q_id, {}).get("answer") != "質問あり":
                st.session_state.answers[q_id] = {"answer": "質問あり", "details": []}
            st.session_state.answers[q_id]['details'].append(user_input)
            with st.spinner("回答を生成しています..."): ai_response = answer_general_question(user_input)
            follow_up_message = f"{ai_response}\n\n他に何かご質問はありますか？"
            st.session_state.messages.append({"role": "assistant", "content": follow_up_message})
    
    elif st.session_state.stage == "asking_main":
        # (このブロックは変更なし)
        response_type = current_q_info["response_type"]
        if response_type == "dialogue":
            st.session_state.stage = "dialogue"; process_answer(user_input, **kwargs); return
        
        answer_type = ""
        if response_type in ["yes_no", "yes_no_detail"]:
            answer_type = "yes" if user_input == "はい" else "no"
        elif response_type == "exam_type":
            answer_type = kwargs.get("answer_type", "unknown")
            
        st.session_state.answers[q_id] = {"answer": user_input, "type": answer_type}

        if response_type == "yes_no":
            if answer_type == "no":
                if current_q_info.get("end_if_no", False):
                    st.session_state.stage = "finished"
                    st.session_state.messages.append({"role": "assistant", "content": "承知いたしました。本日は検査を受けられないとのことですので、問診を終了します。"})
                else:
                    if "follow_up_on_no" in current_q_info: st.session_state.messages.append({"role": "assistant", "content": current_q_info["follow_up_on_no"]})
                    move_to_next_question()
            else: # yes
                if "follow_up_on_yes" in current_q_info: st.session_state.messages.append({"role": "assistant", "content": current_q_info["follow_up_on_yes"]})
                move_to_next_question()
        elif response_type == "yes_no_detail":
            if answer_type == "yes":
                st.session_state.stage = "asking_detail"
                st.session_state.messages.append({"role": "assistant", "content": current_q_info["detail_question_text"]})
            else: move_to_next_question()
        elif response_type == "exam_type": move_to_next_question()
            
# --- UI描画（変更なし） ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if st.session_state.stage != "finished":
    q_index = st.session_state.q_index
    current_q_info = QUESTIONS[q_index]
    response_type = current_q_info.get("response_type")
    input_key = f"input_{current_q_info['id']}_{st.session_state.stage}"

    if st.session_state.stage == "asking_detail":
        if user_input := st.chat_input("詳細を回答してください", key=input_key): process_answer(user_input); st.rerun()
    elif st.session_state.stage in ["asking_main", "dialogue"]:
        if response_type in ["yes_no", "yes_no_detail"]:
            col1, col2 = st.columns(2)
            if col1.button("はい", use_container_width=True, key=f"yes_{input_key}"): process_answer("はい"); st.rerun()
            if col2.button("いいえ", use_container_width=True, key=f"no_{input_key}"): process_answer("いいえ"); st.rerun()
        elif response_type == "exam_type":
            col1, col2, col3 = st.columns(3)
            if col1.button("造影検査", use_container_width=True, key=f"contrast_{input_key}"): process_answer("造影検査", answer_type="contrast"); st.rerun()
            if col2.button("単純検査", use_container_width=True, key=f"simple_{input_key}"): process_answer("単純検査", answer_type="simple"); st.rerun()
            if col3.button("わからない", use_container_width=True, key=f"unknown_{input_key}"): process_answer("わからない", answer_type="unknown"); st.rerun()
        elif response_type == "dialogue":
            if user_input := st.chat_input("質問がある場合は入力してください", key=input_key): process_answer(user_input); st.rerun()
            if st.button("特にありません（問診を終了）", use_container_width=True, key=f"end_dialogue_{input_key}"): process_answer("特にありません"); st.rerun()
else: st.success("問診はすべて終了しました。")