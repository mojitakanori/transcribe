from google.cloud import vision
import io
import os
from openai import OpenAI

# 環境変数からAPIキーを読み込む
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise Exception("APIキーが環境変数 'OPENAI_API_KEY' から取得できませんでした。")

client = OpenAI(api_key=openai_api_key)

# ChatGPTを使って文章を整形するかどうかを設定するフラグ
use_chatgpt = True  # Trueで使用、Falseで使用しない

# ChatGPT APIを使用して文章を整形する関数
def refine_text_with_chatgpt(text):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reviews and corrects text while ensuring the original meaning is preserved. Do not add any new content that is not present in the original text."},
                {"role": "user", "content": f"Please review the following text, correct any grammatical mistakes, improve readability, and fix any incomplete or awkward sections without changing the original meaning:\n\n{text}"}
            ]
        )
        # ChatCompletionMessage オブジェクトの正しいプロパティを参照
        refined_text = completion.choices[0].message.content
        return refined_text
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return text

# Google Cloud Vision APIのクライアントを作成
client_vision = vision.ImageAnnotatorClient()

# 画像フォルダのパスを指定
image_folder_path = r"images"  # 画像フォルダのパスを指定
output_file_path = r"text.txt"  # 出力するテキストファイルのパスを指定

# フォルダ内の画像ファイルを取得
image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 総画像数
total_images = len(image_files)
completed_images = 0

# 10枚ごとにまとめたテキスト
batch_text = ""

# 出力ファイルを開いて書き込みモードに設定
with open(output_file_path, 'w', encoding='utf-8') as output_file:

    # 画像ごとに処理を行う
    for image_index, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder_path, image_file)

        # 画像を読み込む
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        # Vision API用に画像データを設定
        image = vision.Image(content=content)

        # 画像内のテキストを検出
        response = client_vision.text_detection(image=image)
        texts = response.text_annotations

        # テキストが存在する場合のみ処理を行う
        if texts:
            # 抽出されたテキスト（最初の結果が全文）
            raw_text = texts[0].description
            # 空白は削除し、改行はそのまま保持
            clean_text = raw_text.strip()
            # 10枚分のテキストをバッチとしてまとめる
            batch_text += clean_text + "\n\n"

        else:
            batch_text += "テキストが検出されませんでした。\n\n"

        # 処理済みの画像数を増やす
        completed_images += 1

        # 進捗を棒グラフとして表示
        progress_bar = "#" * completed_images + "-" * (total_images - completed_images)
        print(f"[{progress_bar}] {completed_images}/{total_images} images processed")

        # 10枚ごとにChatGPTに送信して整形する（フラグがTrueの場合のみ）
        if completed_images % 10 == 0 or completed_images == total_images:
            if use_chatgpt:
                print(f"ChatGPTに送信中")
                refined_text = refine_text_with_chatgpt(batch_text)
                output_file.write(refined_text)
            else:
                output_file.write(batch_text)

            # 次のバッチのためにテキストをクリア
            batch_text = ""

print(f"テキストが {output_file_path} に保存されました。")
