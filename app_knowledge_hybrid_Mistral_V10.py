import streamlit as st
from llama_cpp import Llama
import json
import copy
import streamlit.components.v1 as components

# --- 定数設定 ---
MODEL_PATH = "./Mistral-Nemo-Japanese-Instruct-2408-Q8_0.gguf"

# --- 専門知識ナ-レッジベース (FAQ) ---
KNOWLEDGE_BASE = {
    "t1_強調画像": {"keywords": ["t1", "t1強調画像"], "answer": "T1強調画像は、組織ごとのT1値（縦緩和時間）の違いを画像にしたものです。一般的に、脂肪組織や造影剤が集まった部分などが白く（高信号に）描出される特徴があります。解剖学的な構造を把握するのに優れています。"},
    "t2_強調画像": {"keywords": ["t2", "t2強調画像"], "answer": "T2強調画像は、組織ごとのT2値（横緩和時間）の違いを画像にしたものです。水成分が多い組織、例えば水腫や嚢胞、炎症などが白く（高信号に）描出される特徴があります。病変を見つけやすい撮影方法の一つです。"},
    "flair": {"keywords": ["flair", "フレア"], "answer": "FLAIR（フレア）は、T2強調画像の一種で、脳脊髄液などの自由水からの信号を抑制（黒く描出）する技術です。これにより、脳室の近くにある病変などが、脳脊髄液の白い信号に紛れることなく明瞭に観察できます。"},
    "拡散強調画像": {"keywords": ["dwi", "拡散", "拡散強調画像"], "answer": "拡散強調画像（DWI）は、組織内の水分子の動き（拡散）を画像化したものです。脳梗塞の超急性期など、細胞のむくみが起きて水分子の動きが制限されると、その部分が白く（高信号に）描出されます。脳梗索の早期診断に非常に有用です。"},
    "造影剤": {"keywords": ["造影剤", "ガドリニウム"], "answer": "MRIで使用される造影剤は、主にガドリニウム製剤です。血管や特定の病変に集まる性質があり、これを使用することで、T1強調画像で病変の範囲や血流の状態がより鮮明にわかります。これにより、診断の精度を高めることができます。"},
    "mra": {"keywords": ["mra", "mrアンгио"], "answer": "MRA（MR Angiography）は、MRIの技術を使って脳や首などの血管だけを立体的に描出する検査です。造影剤を使わずに撮影できる方法と、造影剤を使ってより鮮明に撮影する方法があります。動脈瘤や血管の狭窄・閉塞などを評価するのに用いられます。"},
    "mrs": {"keywords": ["mrs", "mrスペクトロスコピー"], "answer": "MRS（MRスペクトロスコピー）は、MR装置を用いて生体内の特定の代謝物質の濃度を測定する技術です。画像としてではなく、スペクトルとして表示され、脳腫瘍の種類判別などに役立ちます。非侵襲的に生体内の代謝情報を得られる点が特徴です。"},
    "perfusion": {"keywords": ["perfusion", "灌流画像", "かんりゅう"], "answer": "パーフュージョン（灌流画像）は、組織の血流状態を評価するMRIの技術です。造影剤の注入と同時に連続的に画像を取得し、脳血流量（CBF）や脳血流時間（CBV）などの情報を得ます。脳梗塞の急性期における虚血半影（penumbra）の評価などに有用です。"},
    "swi": {"keywords": ["swi", "susceptibility weighted imaging", "susceptibility"], "answer": "SWIは、磁化率効果を利用して、出血、血管、鉄沈着などの情報を強調して描出する撮像法です。微小出血や海綿状血管腫、脳内の石灰化などを高感度で検出できます。特に脳外傷後の微細な出血の検出に優れています。"},
    "dti": {"keywords": ["dti", "拡散テンソル画像"], "answer": "DTI（拡散テンソル画像）は、水分子の拡散方向の異方性を利用して、脳内の神経線維の走行を可視化する撮像法です。神経線維の損傷や脱髄などを評価でき、脳損傷、神経変性疾患などの研究・診断に用いられます。"},
    "t2_star": {"keywords": ["t2*", "t2スター", "ティーツースター"], "answer": "T2*（ティーツースター）強調画像は、ごく微量の出血や鉄の沈着、石灰化などを非常に検出しやすくした撮像法です。これらの部分は信号を失って黒く写ります。SWIという撮像法の元になる画像でもあります。"},
    "asl": {"keywords": ["asl", "動脈スピン標識"], "answer": "ASLは、造影剤を使わずに脳の血流状態を評価できる特殊な撮像法です。患者さん自身の血液を「天然の造影剤」のように利用する技術で、体に負担なく血流の情報を得られる利点があります。"},
    "脂肪抑制": {"keywords": ["脂肪抑制", "fat sat", "ファットサット", "chess"], "answer": "脂肪抑制（Fat Suppression）は、体内の脂肪からの信号を消して、その下にある病変や炎症をより見やすくするための技術です。様々な撮像法と組み合わせて使われる、非常に一般的な手法です。"},
    "fmri": {"keywords": ["fmri", "機能的mri", "ファンクショナル"], "answer": "fMRI（機能的MRI）は、人が何かを考えたり、見たり、話したりしている最中の脳の活動部位を可視化する検査です。脳のどの部分がどのような機能を持っているかを調べるために用いられ、主に脳外科手術前の機能評価などに利用されます。"},
    "ciss": {"keywords": ["ciss", "fiesta", "シス", "フィエスタ"], "answer": "CISSやFIESTAと呼ばれる撮像法は、非常に高い解像度で、脳の中の神経や、内耳、脳脊髄液といった細かい部分を鮮明に描き出すことに特化した技術です。三叉神経痛や顔面痙攣の原因検索、聴神経腫瘍の評価などに用いられます。"},
    "テスラ": {"keywords": ["テスラ", "磁場強度", "1.5t", "3t", "3.0t"], "answer": "テスラ（T）は、MRI装置が持つ磁石の力の強さ（磁場強度）を表す単位です。一般的に1.5テスラや3.0テスラといった装置が使われます。磁場強度が高いほど、より高精細な画像を短時間で得られる傾向にありますが、検査の目的によって適切な装置が選択されます。"},
    "sar": {"keywords": ["sar", "比吸収率", "熱", "あつい"], "answer": "SAR（比吸収率）は、MRI検査中に体内へ吸収される電波のエネルギー量を示す指標です。体温の上昇に関わるため、国際的な基準値が定められています。装置は常にこのSARを監視し、安全な範囲を超えないように自動で制御しながら検査を行っていますのでご安心ください。"},
    "検査の音": {"keywords": ["音", "うるさい", "工事現場", "ガンガン"], "answer": "MRI検査では「ガンガン」「コンコン」といった工事現場のような大きな音がします。これは、画像を作るために装置内部のコイルが高速で振動するために発生する音です。検査中は耳栓やヘッドホンを装着していただきますので、ご安心ください。"},
    "息止め": {"keywords": ["息止め", "呼吸", "お腹", "肝臓"], "answer": "お腹や胸の検査では、撮影中に息を止めていただくことがあります。呼吸で内臓が動くと画像がブレてしまうためです。「息を吸って、吐いて、止めてください」といったマイクからの合図に合わせてご協力をお願いいたします。"},
    "コイル": {"keywords": ["コイル", "受信コイル", "アンテナ"], "answer": "コイルは、体から出てくる微弱な信号を受け取るための、高性能なアンテナ（受信機）のようなものです。検査したい部分（頭、お腹、膝など）に専用のコイルをぴったりと装着することで、より鮮明で綺麗な画像を得ることができます。"},
    "閉所恐怖症": {"keywords": ["狭い", "閉所恐怖症", "苦しい", "パニック"], "answer": "MRI検査は狭いトンネルの中で行いますが、ご気分の悪くなった際には、緊急ブザーですぐにスタッフを呼ぶことができます。また、スタッフは常にマイクやカメラで様子を確認しています。ご不安な点があれば、いつでもお声がけください。"},
    "体動": {"keywords": ["動く", "動いてはいけない", "体動"], "answer": "検査中は、できるだけ体を動かさないようにお願いしています。少しでも動いてしまうと、写真がブレてしまい、正確な診断が難しくなることがあります。綺麗な画像を得るために、リラックスした状態で安静にご協力ください。"},
    "緊急ブザー": {"keywords": ["ブザー", "緊急", "気分が悪い", "ナースコール"], "answer": "検査中は、手にボールのような緊急ブザーをお持ちいただきます。検査中に気分が悪くなったり、何か異常を感じたりした際には、ためらわずにそのブザーを握ってください。すぐに撮影を中断し、スタッフが対応しますのでご安心ください。"},
    "信号強度": {"keywords": ["高信号", "白い", "低信号", "黒い"], "answer": "MRI画像では、組織の種類や状態によって、白っぽく見えたり黒っぽく見えたりします。医師が説明する「高信号」とは画像上で白く見える部分、「低信号」とは黒く見える部分を指します。この白黒の差（コントラスト）から病変を見つけ出します。"},
    "撮像断面": {"keywords": ["断面", "輪切り", "axial", "sagittal", "coronal"], "answer": "MRIでは体を様々な方向から「輪切り」にして観察します。この断面の向きのことを撮像断面と呼びます。頭を上から見た輪切りが「水平断 (Axial)」、体を前から縦に切るのが「冠状断 (Coronal)」、体を横から縦に切るのが「矢状断 (Sagittal)」です。"},
    "stir": {"keywords": ["stir", "スター"], "answer": "STIR（スター）は、脂肪抑制技術の一つで、特に骨や関節、筋肉などの整形外科領域の検査で広く使われます。脂肪の信号を強力に抑制することで、骨の中の浮腫（水が溜まった状態）や炎症、腫瘍などを非常に見つけやすくすることができます。"},
    "vibe_lava": {"keywords": ["vibe", "lava", "ヴァイブ", "ラヴァ"], "answer": "VIBEやLAVAは、主にお腹の造影検査で用いられる、非常に高速な撮影技術のことです。短い息止めの時間で、肝臓や膵臓といった臓器の血流の状態を、ダイナミックに（時間経過を追って）詳しく評価することが可能です。"},
    "金属が危険な理由": {"keywords": ["金属", "なぜダメ", "危険な理由"], "answer": "MRI装置は非常に強力な磁石でできているため、金属には主に３つの危険があります。①金属が磁石に引き寄せられて飛んでしまう「吸引」、②電波の影響で金属が熱くなる「発熱」、③心臓ペースメーカーなどが誤作動を起こす「機能障害」です。安全のため、金属類の持ち込みを厳しく制限しています。"},
    "タトゥーのリスク": {"keywords": ["タトゥー なぜ", "アートメイク なぜ"], "answer": "タトゥーやアートメイクのインクには、酸化鉄などの金属成分が含まれていることがあります。この金属成分がMRI装置の電波に反応して熱を持ち、皮膚にやけど（熱傷）を引き起こす可能性があるため、注意喚起をさせていただいています。"},
    "機能性肌着のリスク": {"keywords": ["ヒートテック", "機能性肌着"], "answer": "一部の発熱性機能性肌着は、汗などの水分に反応して熱を発生させる特殊な繊維を使用しています。MRI検査で使われる電波がこの繊維に影響を与え、予期せぬ発熱ややけどにつながる可能性があるため、検査着へのお着替えをお願いしています。"},
    "mrcp": {"keywords": ["mrcp", "胆管", "膵管"], "answer": "MRCPは、MRIを使って胆嚢（たんのう）や、肝臓から十二指腸へつながる胆管（たんかん）、膵臓の中にある膵管（すいかん）だけを、水を強調する技術で映し出す検査です。造影剤や内視鏡を使わずに、胆石や腫瘍などを調べることができます。"},
    "mrマンモグラフィ": {"keywords": ["マンモグラフィ", "乳房", "乳腺"], "answer": "MRマンモグラフィは、乳房専用のコイルを使って行うMRI検査です。うつ伏せの状態で検査を行い、特に造影剤を用いることで、通常のマンモグラフィや超音波検査では分かりにくい小さな病変を見つけるのに役立ちます。"},
    "心臓mri": {"keywords": ["心臓", "cardiac", "シネ"], "answer": "心臓MRIは、心臓の動き（拍動）や血流、心筋の組織の状態を詳しく調べる検査です。心電図と同期させて撮影することで、ブレのない鮮明な画像を得ます。心筋梗塞や心筋症などの診断に非常に有用です。"},
    "mip": {"keywords": ["mip", "シップ", "最大値投影法"], "answer": "MIP（ミップ）処理は、MRAなどで撮影した3次元の血管データから、血管部分だけを抽出して立体的な画像を作成する技術の一つです。医師が血管の走行や狭窄、動脈瘤などを直感的に把握しやすくするために用いられます。"},
    "mpr": {"keywords": ["mpr", "多断面再構成"], "answer": "MPRは、一度撮影した3Dデータをもとに、コンピュータ上で後から自由な角度の断面図を作り出す技術です。例えば、最初に輪切りの画像を撮影しておけば、後から縦や斜めの断面図を自由に作成でき、病変を様々な角度から詳細に観察できます。"},
    "鎮静": {"keywords": ["鎮静", "眠る", "子供", "小児"], "answer": "小さなお子様や、狭い場所がどうしても苦手で検査中に安静を保つのが難しい方のために、お薬を使って少しうとうとした状態で検査を行うことがあります。これを鎮静といいます。医師の判断のもと、安全に配慮して行われます。"},
    "アーチファクト": {"keywords": ["アーチファクト", "画像の乱れ", "ゴースト"], "answer": "アーチファクトとは、体の動きや体内の金属、装置の不具合など、様々な原因によって生じる画像の乱れや偽の像（ゴーストなど）のことです。正確な診断の妨げになるため、検査中は体を動かさないなどのご協力をお願いしています。"}
}

# --- 問診項目テンプレート ---
QUESTIONS = [
    {"id": "q1", "text": "これからMRI検査を受けられますか？", "response_type": "yes_no", "end_if_no": True},
    {"id": "q_exam_type", "text": "これから受けられる検査は、造影剤を使用する「**造影検査**」ですか？それとも造影剤を使用しない「**単純検査**」ですか？\n\nもし不明な場合は「わからない」を選択してください。", "response_type": "exam_type"},
    {"id": "q_asthma", "text": "**気管支喘息**と診断されたことはありますか？", "response_type": "yes_no", "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_yes": "ご申告ありがとうございます。安全のため、後ほど**担当者が改めて確認**させていただきます。"},
    {"id": "q_allergy", "text": "今までに薬や食べ物で**アレルギー** (発疹、かゆみ、気分不快など) を起こしたことはありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。差し支えなければ、**何に対するアレルギーか、該当するものをすべて**教えていただけますか？\n（選択するとボタンの色が変わります）", "detail_choices": ["薬", "食べ物", "造影剤", "金属"], "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_detail": "重要な情報をありがとうございます。"},
    {"id": "q_kidney", "text": "以前に**腎臓の病気**を指摘されたことはありますか？", "response_type": "yes_no", "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_yes": "ご申告ありがとうございます。安全のため、後ほど**担当者が改めて確認**させていただきます。"},
    {"id": "q_contrast_side_effect", "text": "以前に MRI や CT の造影検査で**気分が悪くなったり、副作用が出たりした**ことはありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。**どのような症状が出たか、該当するものをすべて**教えていただけますか？\n（選択するとボタンの色が変わります）", "detail_choices": ["吐き気・気分不快", "じんましん・かゆみ", "息苦しさ"], "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_detail": "詳細な情報をありがとうございます。こちらも安全のため、後ほど**改めて確認**させていただきます。"},
    {"id": "q_consent_form", "text": "「MRI 造影検査 **同意書兼問診票**」に、ご自身で署名されましたか？", "response_type": "yes_no", "skip_if": {"q_id": "q_exam_type", "answer_type": "simple"}, "follow_up_on_no": "承知いたしました。安全な検査のために**同意書の署名は必須**となります。後ほど**担当者が改めてご案内・確認**いたしますのでご安心ください。"},
    {"id": "q_previous_mri", "text": "以前に MRI 検査を受けたことはありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。その検査について、**「部位」**と**「場所」**をボタンで選択してください。", "detail_choices": ["今回と同じ部位", "別の部位"]},
    {"id": "q_internal_metal", "text": "体内に**ペースメーカー等の医療機器**や、**手術で入れた金属**（クリップ、人工関節など）はありますか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。**該当するものをすべて**ボタンで選択してください。\n（選択するとボタンの色が変わります）", "detail_choices": ["心臓ペースメーカー/ICD", "脳動脈瘤クリップ/コイル", "人工関節/ボルト", "人工内耳/刺激装置", "シャント/測定装置", "不明な金属"]},
    {"id": "q_beauty", "text": "以下の項目のうち、現在使用しているものはありますか？ **該当するものをすべて**ボタンで選択してください。\n（選択するとボタンの色が変わります）", "response_type": "multiple_choice", "choices": ["マニキュア/ネイル", "マスカラ/アイシャドウ", "増毛パウダー", "何もない"], "follow_up": "ご申告ありがとうございます。こちらも安全のため、後ほど**改めて確認**させていただきます。"},
    {"id": "q_jewel_nail", "text": "宝石や金属の飾りがついたネイル（**ジュエルネイル**）ですか？", "response_type": "yes_no", "skip_unless": {"q_id": "q_beauty", "answer_contains": "マニキュア/ネイル"}, "follow_up_on_yes": "ご申告ありがとうございます。ネイルの素材によっては、検査中に熱を持ったり、画像の乱れの原因となる可能性があります。安全のため、後ほど**担当者が直接確認**させていただきます。"},
    {"id": "q_tattoo", "text": "**刺青**や**タトゥー**・**アートメイク**をしていませんか？", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。**どちらをされていますか？**", "detail_choices": ["刺青/タトゥー", "アートメイク"], "follow_up_on_detail": "重要な情報をありがとうございます。安全のため、後ほど**担当者が改めて確認**させていただきます。"},
    {"id": "q_accessories", "text": """**装飾品**や**持ち物**に関する最終確認です。\n**検査室に持ち込めないもの**がありますので、以下をご確認ください。\n\n- **金属・アクセサリー類:**\n  - 指輪、ネックレス、ピアス、ヘアピン、時計\n- **衣類・装着物:**\n  - 金属のワイヤーやホック付きの下着\n  - 保温下着（ヒートテックは除く）、コルセット\n  - 湿布、カイロ、エレキバン、貼り薬（ニトロダーム等）\n- **その他:**\n  - 入れ歯、補聴器、義眼、ウィッグ、カラーコンタクト\n  - 携帯電話、スマートフォン、磁気カード、鍵\n\n上記のようなものを、現在身につけていたり、お持ちですか？""", "response_type": "yes_no_detail", "detail_question_text": "承知いたしました。**該当するものをすべて**ボタンで選択してください。\n（選択するとボタンの色が変わります）", "detail_choices": ["指輪/ネックレス/ピアス", "時計/ヘアピン", "入れ歯/補聴器", "湿布/カイロ/貼り薬", "金属付き下着/コルセット", "携帯/カード/鍵"], "follow_up_on_detail": "ご申告ありがとうございます。検査室に入る前に、**スタッフが改めてご案内します**のでご安心ください。"},
    {"id": "q_claustrophobia", "text": "現在、**狭い場所でじっとしていて苦しくなる**ことがありませんか？", "response_type": "yes_no", "follow_up_on_yes": "ご申告ありがとうございます。検査中は**スタッフが常にマイクやカメラで様子を見ており、いつでも会話ができます**。また、ご気分が悪くなった際には、**緊急ブザーでお知らせいただけます**のでご安心ください。"},
    {"id": "q_pregnancy", "text": "現在、**妊娠されている可能性**はありますか？", "response_type": "yes_no", "follow_up_on_yes": "ご申告ありがとうございます。妊娠中のMRI検査は慎重な判断が必要なため、後ほど**担当者が改めて詳細をお伺い**いたします。"},
    {"id": "q_other_concerns", "text": "他に何か、ご心配なことやご質問はありますか？", "response_type": "dialogue"},
]

def scroll_to_end():
    js = f"""<script>
        function scrollToEnd() {{
            setTimeout(() => {{
                try {{
                    const endOfChat = window.parent.document.getElementById('end-of-chat');
                    if (endOfChat) {{ endOfChat.scrollIntoView({{ behavior: 'smooth', block: 'end' }}); }}
                }} catch (e) {{ console.error("Scroll to end failed:", e); }}
            }}, 100);
        }}
        scrollToEnd();
    </script>"""
    components.html(js, height=0, scrolling=False)
@st.cache_resource
def load_model():
    try: return Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, verbose=False, chat_format="mistral-instruct")
    except Exception as e: st.error(f"モデルのロードに失敗しました: {e}\nMODEL_PATHが正しいか確認してください。"); st.stop()
llm = load_model()
def classify_user_intent(user_input):
    prompt = f"""[INST] <<SYS>>\nあなたはテキスト分類アシスタントです。患者の発言の意図を「質問・懸念あり(question)」か「特にない・終了(end)」に分類し、JSON形式で{{"intent": "..."}}とだけ出力してください。\n<</SYS>>\n患者の発言: {user_input}\n[/INST]"""
    try:
        output = llm(prompt, max_tokens=32, temperature=0.0)
        response_text = output["choices"][0]["text"]
        start_index = response_text.find('{'); end_index = response_text.rfind('}') + 1
        if start_index != -1 and end_index != -1: json_str = response_text[start_index:end_index]; data = json.loads(json_str); return data.get("intent", "question")
        return "question"
    except Exception: return "question"
def find_knowledge(user_question):
    q_lower = user_question.lower()
    for key, data in KNOWLEDGE_BASE.items():
        for keyword in data["keywords"]:
            if keyword in q_lower: return data["answer"]
    return None
def answer_general_question(user_input):
    q_lower = user_input.lower()
    time_keywords = ["時間", "どのくらい", "何分", "かかりますか"]
    if "検査" in q_lower and any(keyword in q_lower for keyword in time_keywords):
        exam_type_info = st.session_state.answers.get("q_exam_type", {}); exam_type = exam_type_info.get("answer", "わからない")
        if exam_type == "単純検査": return "お答えします。単純検査の場合、検査時間は15分から20分程度が目安です。検査部位や撮影内容によって多少前後することがあります。"
        elif exam_type == "造影検査": return "お答えします。造影検査の場合、検査時間は15分から25分程度が目安です。造影剤の注入や撮影の追加があるため、単純検査より少し長くなることがあります。"
        else: return "お答えします。検査時間は検査の種類によって異なります。造影剤を使わない「単純検査」の場合は15分から20分程度、造影剤を使う「造影検査」の場合は15分から25分程度が目安となります。"
    found_knowledge = find_knowledge(user_input)
    if found_knowledge:
        prompt = f"""[INST] <<SYS>>\nあなたは、提供された専門知識を元に、患者さんからの質問に分かりやすく回答する医療アシスタントです。あなたの役割は、与えられたコンテキスト（専門知識）の内容を逸脱せず、それを要約・整形して、親切な言葉で伝えることです。絶対にコンテキストに書かれていない情報を付け加えてはいけません。\n<</SYS>>\n# 専門知識コンテキスト\n{found_knowledge}\n# 患者の質問\n{user_input}\n# 指示\n上記の「専門知識コンテキスト」だけを情報源として、患者の質問に1〜3文程度の平易な言葉で回答してください。\n[/INST]"""
        try: output = llm(prompt, max_tokens=512, temperature=0.1, stop=["</s>", "[INST]"]); return output["choices"][0]["text"].strip()
        except Exception as e: st.error(f"LLMエラー（ナレッジベース）: {e}"); return f"申し訳ありません、回答の生成中にエラーが発生しました。いただいたご質問に関する情報として、以下を共有いたします。\n\n---\n{found_knowledge}\n\n---\n詳細は当日担当者にご確認ください。"
    else:
        prompt = f"""[INST] <<SYS>>\nあなたはMRI検査に関する質問に対応する、誠実で慎重なAIアシスタントです。あなたの役割は、患者の質問が「誰にでも共通する一般的な内容」であるか、それとも「専門的・医学的な内容」であるかを厳密に判断することです。\n# 重要ルール\n- **少しでも専門的、医学的、または個別性の高い内容が含まれる場合、絶対に回答を創作せず、正直に「回答不可能」と判断してください。**\n- あなたの役割は知っていることを話すことではなく、安全に回答できる範囲を見極めることです。\n# 回答可能な質問の例\n- 検査当日の流れ、食事制限、服装、検査中の大きな音、閉所感への対策など。例：「検査の前に食事はできますか？」\n# 回答不可能な質問の例（これらの場合は必ず "answerable": false としてください）\n- MRIの撮像原理や専門用語、個人の病状や診断、治療方針、検査結果の解釈など。\n# 出力形式\n- 回答可能な場合: `{{"answerable": true, "response": "ここに患者への回答を記述します。親切で分かりやすい言葉を選び、1〜3文程度で簡潔にまとめてください。"}}`\n- 回答不可能な場合: `{{"answerable": false, "response": "お尋ねの件は、専門的な知識が必要なご質問となります。不正確な情報をお伝えすることを避けるため、私からお答えすることはできません。患者様の安全を第一に考え、いただいたご質問は担当者に正確に申し送り、当日、医師や技師から直接ご説明させていただきますので、ご安心ください。"}}`\n<</SYS>>\n患者の質問: {user_input}\n[/INST]"""
        try:
            output = llm(prompt, max_tokens=512, temperature=0.1, stop=["</s>", "[INST]"]); response_text = output["choices"][0]["text"].strip()
            start_index = response_text.find('{'); end_index = response_text.rfind('}') + 1
            if start_index != -1 and end_index != -1: json_str = response_text[start_index:end_index]; data = json.loads(json_str); return data.get("response", "申し訳ありません、現在回答を準備できません。担当者へ確認事項として申し伝えます。")
            else: return "申し訳ありません、回答の形式に問題がありました。担当者へ確認事項として申し伝えます。"
        except Exception as e: st.error(f"LLMエラー（一般質問）: {e}"); return "申し訳ありません、現在回答を生成できません。この質問は担当者に申し送りいたします。"

# --- Streamlit App ---
# 変更点: カラムの比率を [4, 1] から [7, 2] に変更し、両方のカラムに適切な幅を割り当てます。
col_title, col_settings = st.columns([7, 2])
with col_title:
    st.title("MRI AI 問診 🏥")
    st.caption("AIとルールベースのハイブリッドで、安定かつ柔軟な問診を実現します。")
with col_settings:
    st.write("文字サイズ")
    font_size_label = st.radio("文字サイズ", ["標準", "大きく", "さらに大きく"], key="font_size_selection", horizontal=True, label_visibility="collapsed")

font_sizes = {"標準": "16px", "大きく": "18px", "さらに大きく": "20px"}
selected_size = font_sizes.get(st.session_state.font_size_selection, "16px")
st.markdown(f"""
<style>
html, body, [class*="st-"], .st-emotion-cache-1kyxreq e1nzilvr5, .st-emotion-cache-1kyxreq e1nzilvr5 p {{ font-size: {selected_size} !important; }}
div[data-testid="stButton"] > button {{ height: auto !important; padding-top: 0.75em !important; padding-bottom: 0.75em !important; }}
.block-container {{ padding-top: 2rem; }}
</style>
""", unsafe_allow_html=True)

if "stage" not in st.session_state:
    st.session_state.stage = "asking_main"; st.session_state.q_index = 0; st.session_state.answers = {}; st.session_state.messages = []; st.session_state.history = []
    if "font_size_selection" not in st.session_state: st.session_state.font_size_selection = "標準"
    first_q = QUESTIONS[0]["text"]; initial_message = f"こんにちは。MRI検査を安全に行うため、いくつか質問をさせていただきます。\n\n{first_q}"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

def go_back():
    if st.session_state.history:
        last_state = st.session_state.history.pop()
        st.session_state.q_index = last_state["q_index"]; st.session_state.answers = last_state["answers"]; st.session_state.messages = last_state["messages"]; st.session_state.stage = last_state["stage"]

def move_to_next_question():
    current_q_index = st.session_state.q_index; st.session_state.q_index += 1
    while st.session_state.q_index < len(QUESTIONS):
        next_q_info = QUESTIONS[st.session_state.q_index]; should_skip = False
        if "skip_if" in next_q_info:
            condition = next_q_info["skip_if"]
            if (condition["q_id"] in st.session_state.answers and st.session_state.answers[condition["q_id"]]["type"] == condition["answer_type"]): should_skip = True
        if "skip_unless" in next_q_info:
            condition = next_q_info["skip_unless"]
            q_id = condition["q_id"]
            previous_answer_data = st.session_state.answers.get(q_id, {})
            previous_answer = previous_answer_data.get("answer", [])

            is_condition_met = False
            if isinstance(previous_answer, list):
                if "answer_contains" in condition:
                    if condition["answer_contains"] in previous_answer:
                        is_condition_met = True
                elif "answer_contains_any" in condition:
                    if any(req in previous_answer for req in condition["answer_contains_any"]):
                        is_condition_met = True
            
            if not is_condition_met:
                should_skip = True

        if should_skip: st.session_state.q_index += 1; continue
        else:
            current_response_type = QUESTIONS[current_q_index].get('response_type')
            if current_response_type == 'dialogue': message = f"{next_q_info['text']}"
            else: message = f"承知いたしました。では次の質問です。\n\n{next_q_info['text']}"
            st.session_state.messages.append({"role": "assistant", "content": message}); return
    st.session_state.stage = "finished"; st.session_state.messages = []
    
    noteworthy_items = []; other_items = []
    yes_is_noteworthy_qids = ["q_asthma", "q_allergy", "q_kidney", "q_contrast_side_effect", "q_internal_metal", "q_jewel_nail", "q_tattoo", "q_accessories", "q_claustrophobia", "q_pregnancy"]
    for q_info in QUESTIONS:
        q_id = q_info["id"]
        if q_id in st.session_state.answers:
            answer_data = st.session_state.answers[q_id]; answer = answer_data.get('answer', [])
            if isinstance(answer, list): answer_text = ", ".join(answer) if answer else "なし"
            else: answer_text = answer
            answer_data['display_text'] = answer_text
            is_noteworthy = False
            if isinstance(answer, list) and answer and not (len(answer) == 1 and answer[0] == '何もない'): is_noteworthy = True
            elif answer_text == "はい" and q_id in yes_is_noteworthy_qids: is_noteworthy = True
            elif answer_text == "いいえ" and q_id in ["q1", "q_consent_form"]: is_noteworthy = True
            elif q_id == "q_other_concerns" and answer_text == "質問あり": is_noteworthy = True
            if is_noteworthy: noteworthy_items.append((q_info, answer_data))
            else: other_items.append((q_info, answer_data))
    
    final_summary = "ご協力ありがとうございました。以下が問診のサマリーです。\n\n---"
    if noteworthy_items:
        final_summary += "\n\n**【⚠️ 要確認項目】**\n"
        for q_info, answer_data in noteworthy_items:
            answer_text = answer_data.get('display_text', '（回答なし）'); final_summary += f'''\n**Q: {q_info['text']}**\nA: **{answer_text}**'''
            if "detail" in answer_data: final_summary += f'''\n> **詳細:** {answer_data['detail']}\n'''
            elif "details" in answer_data and answer_data.get('details'):
                final_summary += "\n> **追加の質問と回答の履歴:**\n"
                for qa_pair in answer_data.get('details', []):
                    user_q = qa_pair.get('question', '（質問不明）').replace('\n', ' '); ai_a_formatted = qa_pair.get('answer', '（回答不明）').replace('\n', '\n>   '); final_summary += f'''> - **【患者様の質問】** {user_q}\n>   **【AIの回答】** {ai_a_formatted}\n'''
            else: final_summary += "\n"
        final_summary += "\n---"
    if other_items:
        final_summary += "\n\n**【その他の回答項目】**\n"
        for q_info, answer_data in other_items:
            answer_text = answer_data.get('display_text', '（回答なし）'); final_summary += f'''\n**Q: {q_info['text']}**\nA: {answer_text}'''
            if "detail" in answer_data: final_summary += f'''\n> **詳細:** {answer_data['detail']}\n'''
            else: final_summary += "\n"
    if not noteworthy_items and not other_items: final_summary = "ご協力ありがとうございました。\n\n**【問診サマリー】**\n\nすべての問診項目がスキップされたか、回答がありませんでした。"
    st.session_state.messages.append({"role": "assistant", "content": final_summary})

def process_answer(user_input, **kwargs):
    if st.session_state.stage in ["asking_main", "asking_detail"]:
        st.session_state.history.append({"q_index": st.session_state.q_index, "answers": copy.deepcopy(st.session_state.answers), "messages": copy.deepcopy(st.session_state.messages), "stage": st.session_state.stage})
    
    q_index = st.session_state.q_index
    current_q_info = QUESTIONS[q_index]
    q_id = current_q_info["id"]

    user_message = user_input
    if isinstance(user_input, list):
        user_message = f"（選択：{', '.join(user_input) if user_input else 'なし'}）"

    st.session_state.messages.append({"role": "user", "content": user_message})

    if st.session_state.stage == "asking_detail":
        if q_id == "q_internal_metal":
            st.session_state.answers[q_id]["detail"] = user_message
            st.session_state.stage = "asking_metal_timing"
            st.session_state.messages.append({"role": "assistant", "content": "承知いたしました。その手術をいつ頃受けられましたか？\n（体内金属の種類によっては、製造時期の確認が安全上非常に重要となります）"})
        else:
            st.session_state.answers[q_id]["detail"] = user_message
            if "follow_up_on_detail" in current_q_info:
                st.session_state.messages.append({"role": "assistant", "content": current_q_info["follow_up_on_detail"]})
            st.session_state.stage = "asking_main"
            move_to_next_question()
    elif st.session_state.stage == "asking_metal_timing":
        st.session_state.answers[q_id]["detail"] += f"、 手術時期: {user_input}"
        st.session_state.messages.append({"role": "assistant", "content": "重要な情報をありがとうございます。安全を最優先するため、後ほど担当者が改めて詳細を確認させていただきますのでご安心ください。"})
        st.session_state.stage = "asking_main"
        move_to_next_question()
    elif st.session_state.stage == "dialogue":
        end_phrases = ["特にありません", "ありません", "ないです", "大丈夫です", "以上です", "ない"]; normalized_input = user_input.strip().replace("。", "").replace("、", "")
        intent = "question";
        if normalized_input in end_phrases: intent = "end"
        else:
            with st.spinner("内容を確認しています..."): intent = classify_user_intent(user_input)
        if intent == "end":
            if not st.session_state.answers.get(q_id): st.session_state.answers[q_id] = {"answer": user_input}
            st.session_state.stage = "asking_main"; move_to_next_question()
        else:
            if st.session_state.answers.get(q_id, {}).get("answer") != "質問あり": st.session_state.answers[q_id] = {"answer": "質問あり", "details": []}
            with st.spinner("回答を生成しています..."): ai_response = answer_general_question(user_input)
            st.session_state.answers[q_id]['details'].append({"question": user_input, "answer": ai_response}); follow_up_message = f"{ai_response}\n\n他に何かご質問はありますか？"; st.session_state.messages.append({"role": "assistant", "content": follow_up_message})
    elif st.session_state.stage == "asking_main":
        response_type = current_q_info["response_type"]
        if response_type == "dialogue": st.session_state.stage = "dialogue"; process_answer(user_input, **kwargs); return
        answer_type = ""
        if response_type in ["yes_no", "yes_no_detail"]: answer_type = "yes" if user_input == "はい" else "no"
        elif response_type == "exam_type": answer_type = kwargs.get("answer_type", "unknown")
        st.session_state.answers[q_id] = {"answer": user_input, "type": answer_type}
        if response_type == "multiple_choice":
            is_other_selected = any(item != "何もない" for item in user_input)
            if user_input and is_other_selected and "follow_up" in current_q_info:
                st.session_state.messages.append({"role": "assistant", "content": current_q_info["follow_up"]})
        if response_type == "yes_no":
            if answer_type == "no":
                if current_q_info.get("end_if_no", False): st.session_state.stage = "finished"; st.session_state.messages.append({"role": "assistant", "content": "承知いたしました。本日は検査を受けられないとのことですので、問診を終了します。"}); move_to_next_question()
                else:
                    if "follow_up_on_no" in current_q_info: st.session_state.messages.append({"role": "assistant", "content": current_q_info["follow_up_on_no"]})
                    move_to_next_question()
            else:
                if "follow_up_on_yes" in current_q_info: st.session_state.messages.append({"role": "assistant", "content": current_q_info["follow_up_on_yes"]})
                move_to_next_question()
        elif response_type == "yes_no_detail":
            if answer_type == "yes": st.session_state.stage = "asking_detail"; st.session_state.messages.append({"role": "assistant", "content": current_q_info["detail_question_text"]})
            else: move_to_next_question()
        else: move_to_next_question()

# --- UI描画 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.stage != "finished":
    q_index = st.session_state.q_index; current_q_info = QUESTIONS[q_index]; q_id = current_q_info['id']; response_type = current_q_info.get("response_type"); input_key = f"input_{q_id}_{st.session_state.stage}"

    def multi_select_buttons(choices, q_id):
        temp_key = f'temp_multiselect_buttons_{q_id}';
        if temp_key not in st.session_state: st.session_state[temp_key] = {choice: False for choice in choices}

        def toggle_button_state(choice_to_toggle):
            if choice_to_toggle == "何もない":
                st.session_state[temp_key] = {choice: False for choice in choices}
                st.session_state[temp_key][choice_to_toggle] = True
            else:
                st.session_state[temp_key][choice_to_toggle] = not st.session_state[temp_key][choice_to_toggle]
                if st.session_state[temp_key].get("何もない"):
                    st.session_state[temp_key]["何もない"] = False

        cols = st.columns(2)
        for i, choice in enumerate(choices):
            col_index = i % 2; is_selected = st.session_state[temp_key].get(choice, False); button_type = "primary" if is_selected else "secondary"; unique_key = f"btn_{choice.replace('/', '_').replace(' ', '_')}_{q_id}"
            cols[col_index].button(choice, key=unique_key, on_click=toggle_button_state, args=(choice,), use_container_width=True, type=button_type)
        
        if not st.session_state[temp_key].get("何もない", False):
            other_input_key = f"other_text_{q_id}"; other_info = st.text_input("その他（上記にない場合）:", key=other_input_key)
        else:
            other_info = ""

        return st.session_state[temp_key], other_info


    multi_select_detail_qids = ['q_internal_metal', 'q_accessories', 'q_allergy', 'q_contrast_side_effect']
    if response_type == "multiple_choice" or (st.session_state.stage == "asking_detail" and q_id in multi_select_detail_qids):
        choices = current_q_info.get("choices") or current_q_info.get("detail_choices", [])
        selected_states, other_info = multi_select_buttons(choices, q_id)
        button_cols = st.columns([3, 1]) if st.session_state.history else [1]
        with button_cols[0]:
            if st.button("決定して次へ", key=f"submit_multiselect_{input_key}", use_container_width=True):
                selected_choices = [choice for choice, checked in selected_states.items() if checked]
                if other_info.strip(): selected_choices.append(f"その他: {other_info.strip()}")
                if not selected_choices:
                    st.warning("いずれかの項目を選択してください。")
                else:
                    process_answer(selected_choices); st.rerun()
        if st.session_state.history and len(button_cols) > 1:
            with button_cols[1]: st.button("↩️ 戻る", on_click=go_back, key=f"back_multiselect_{input_key}", use_container_width=True)

    elif st.session_state.stage == "asking_metal_timing":
        main_choices = ["1990年代以前", "2000年代", "2010年代", "2020年代以降"]
        other_choice = "わからない"

        # 2x2の年代選択ボタン
        for i in range(0, len(main_choices), 2):
            cols = st.columns(2)
            # 1列目のボタン
            if cols[0].button(main_choices[i], use_container_width=True, key=f"timing_choice_{i}_{input_key}"):
                process_answer(main_choices[i]); st.rerun()
            # 2列目のボタン (存在する場合)
            if (i + 1) < len(main_choices):
                if cols[1].button(main_choices[i+1], use_container_width=True, key=f"timing_choice_{i+1}_{input_key}"):
                    process_answer(main_choices[i+1]); st.rerun()
        
        st.markdown("---") # 視覚的な区切り線

        # 「わからない」と「戻る」ボタン
        if st.session_state.history:
            cols_bottom = st.columns(2)
            if cols_bottom[0].button(other_choice, use_container_width=True, key=f"timing_choice_unknown_{input_key}"):
                process_answer(other_choice); st.rerun()
            cols_bottom[1].button("↩️ 戻る", on_click=go_back, key=f"back_timing_{input_key}", use_container_width=True)
        else:
            if st.button(other_choice, use_container_width=True, key=f"timing_choice_unknown_{input_key}"):
                process_answer(other_choice); st.rerun()

    elif st.session_state.stage == "asking_detail":
        if q_id == 'q_previous_mri':
            def set_mri_selection(category, value): st.session_state[f"temp_mri_{category}"] = value
            st.markdown("**部位**"); part_cols = st.columns(len(current_q_info["detail_choices"]))
            for i, choice in enumerate(current_q_info["detail_choices"]):
                is_selected = (st.session_state.get("temp_mri_part") == choice)
                part_cols[i].button(choice, key=f"part_{choice}", on_click=set_mri_selection, args=("part", choice), use_container_width=True, type="primary" if is_selected else "secondary")
            
            st.markdown("**場所**"); location_choices = ["当院", "当院以外"]; loc_cols = st.columns(len(location_choices))
            for i, choice in enumerate(location_choices):
                is_selected = (st.session_state.get("temp_mri_location") == choice)
                loc_cols[i].button(choice, key=f"loc_{choice}", on_click=set_mri_selection, args=("location", choice), use_container_width=True, type="primary" if is_selected else "secondary")

            additional_info = st.text_area("もし分れば、いつ頃・どちらの病院で受けたかなどもご記入ください。（任意）", key=f"text_{q_id}")
            button_cols = st.columns([3, 1]) if st.session_state.history else [1]
            with button_cols[0]:
                if st.button("決定して次へ", key=f"submit_custom_{q_id}", use_container_width=True):
                    part = st.session_state.get("temp_mri_part"); location = st.session_state.get("temp_mri_location")
                    if not part or not location: st.warning("「部位」と「場所」の両方を選択してください。")
                    else:
                        detail_answer = f"部位: {part}, 場所: {location}"
                        if additional_info.strip(): detail_answer += f" (補足: {additional_info.strip()})"
                        if "temp_mri_part" in st.session_state: del st.session_state.temp_mri_part
                        if "temp_mri_location" in st.session_state: del st.session_state.temp_mri_location
                        process_answer(detail_answer); st.rerun()
            if st.session_state.history and len(button_cols) > 1:
                with button_cols[1]: st.button("↩️ 戻る", on_click=go_back, key=f"back_custom_{q_id}", use_container_width=True)
        else:
            free_input_key = f"free_input_for_{q_id}";
            if "detail_choices" in current_q_info and not st.session_state.get(free_input_key, False):
                choices = current_q_info["detail_choices"]; cols = st.columns(min(len(choices), 3))
                for i, choice in enumerate(choices):
                    if cols[i % 3].button(choice, key=f"choice_{i}_{input_key}", use_container_width=True): process_answer(choice); st.rerun()
                if st.session_state.history:
                    col_other, col_back = st.columns(2)
                    with col_other:
                        if st.button("その他（自由入力）", key=f"other_{input_key}", use_container_width=True): st.session_state[free_input_key] = True; st.rerun()
                    with col_back:
                        st.button("↩️ 一つ前の質問に戻る", on_click=go_back, key=f"back_detail_choice_{input_key}", use_container_width=True)
                else:
                    if st.button("その他（自由入力）", key=f"other_{input_key}", use_container_width=True): st.session_state[free_input_key] = True; st.rerun()
            else:
                st.write("詳細を自由に入力してください。入力後、**「送信」ボタンを押してください。**")
                col_text, col_buttons = st.columns([3, 1])
                with col_text: user_input_detail = st.text_area("詳細", key=input_key, height=100, label_visibility="collapsed")
                with col_buttons:
                    if st.button("送信", key=f"submit_detail_{input_key}", use_container_width=True):
                        if user_input_detail: process_answer(user_input_detail); st.rerun()
                    if st.session_state.history: st.button("↩️ 戻る", on_click=go_back, key=f"back_detail_free_{input_key}", use_container_width=True)
    elif st.session_state.stage in ["asking_main", "dialogue"]:
        if response_type in ["yes_no", "yes_no_detail"]:
            cols = st.columns([2, 2, 1]) if st.session_state.history else st.columns(2)
            if cols[0].button("はい", use_container_width=True, key=f"yes_{input_key}"): process_answer("はい"); st.rerun()
            if cols[1].button("いいえ", use_container_width=True, key=f"no_{input_key}"): process_answer("いいえ"); st.rerun()
            if len(cols) > 2: cols[2].button("↩️ 戻る", on_click=go_back, key=f"back_{input_key}", use_container_width=True)
        elif response_type == "exam_type":
            cols = st.columns([2, 2, 2, 1]) if st.session_state.history else st.columns(3)
            if cols[0].button("造影検査", use_container_width=True, key=f"contrast_{input_key}"): process_answer("造影検査", answer_type="contrast"); st.rerun()
            if cols[1].button("単純検査", use_container_width=True, key=f"simple_{input_key}"): process_answer("単純検査", answer_type="simple"); st.rerun()
            if cols[2].button("わからない", use_container_width=True, key=f"unknown_{input_key}"): process_answer("わからない", answer_type="unknown"); st.rerun()
            if len(cols) > 3: cols[3].button("↩️ 戻る", on_click=go_back, key=f"back_{input_key}", use_container_width=True)
        elif response_type == "dialogue":
            st.write("質問がある場合は入力してください。日本語入力の確定はEnterではなく、**「送信」ボタンを押してください。**")
            user_input_dialogue = st.text_area("質問", key=input_key, height=100, label_visibility="collapsed")
            cols = st.columns([2, 3, 1]) if st.session_state.history else st.columns(2)
            if cols[0].button("送信", key=f"submit_dialogue_{input_key}", use_container_width=True):
                if user_input_dialogue: process_answer(user_input_dialogue); st.rerun()
            if cols[1].button("特にありません（問診を終了）", use_container_width=True, key=f"end_dialogue_{input_key}_button"): process_answer("特にありません"); st.rerun()
            if len(cols) > 2: cols[2].button("↩️ 戻る", on_click=go_back, key=f"back_{input_key}", use_container_width=True)
else:
    st.success("問診はすべて終了しました。ありがとうございました。")

if st.session_state.stage != "finished":
    st.markdown("<div id='end-of-chat' style='height: 250px;'></div>", unsafe_allow_html=True)
    scroll_to_end()