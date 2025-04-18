import json
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from together import Together

# model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# client = OpenAI(
#     api_key="UNKOWN",
#     base_url="https://ec32-34-138-204-180.ngrok-free.app/v1"
# )

# Load the dataset
df = pd.read_csv('/home/nampq/projects/specialist-prediction/data/data_version3/dataset.csv')
# Check for missing columns
required_columns = ['reason_combind', 'specialist_name']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in the dataset: {', '.join(missing_columns)}")

system_prompt = """
Bạn là hệ thống xử lý văn bản y tế chuyển đổi lý do khám bệnh thành triệu chứng và bệnh tật bằng tiếng Việt.

# Định dạng đầu vào
{
  "records": [
    {
      "reason_combined": "Lý do khám bệnh của bệnh nhân",
      "specialist_name": "Chuyên khoa được tư vấn"
    },
    {...}
  ]
}

# Quy tắc cơ bản
1. Chỉ trích xuất các tình trạng y tế THỰC SỰ đang hiện diện
2. Sửa lỗi chính tả (ví dụ: "bung" → "bụng", "xuyễn" → "suyễn")
3. Định dạng như danh sách phân cách bằng dấu phẩy CHỈ BẰNG TIẾNG VIỆT: "triệu chứng1, triệu chứng2, bệnh1, bệnh2"
4. Trả về null khi không đạt điều kiện trích xuất
5. LUÔN đầu ra bằng tiếng Việt, không bao giờ dịch sang tiếng Anh hoặc ngôn ngữ khác

# Khi nào trả về null
- Triệu chứng/bệnh tật không phù hợp với chuyên khoa
- Không có triệu chứng hiện tại được đề cập (ngoại trừ chuyên khoa "nội khoa")
- Các sự kiện y tế trong quá khứ không có triệu chứng hiện tại ("đã mổ...", "tiền sử...")
- Yêu cầu hành chính, tái khám định kỳ, hoặc yêu cầu hồ sơ
- Kết quả xét nghiệm không có triệu chứng liên quan

# Trường hợp đặc biệt
- Chỉ đối với "nội khoa": "khám tổng quát" là hợp lệ và được giữ nguyên
- Không bao giờ thay đổi ý nghĩa y tế - chỉ sửa lỗi chính tả nhưng giữ nguyên tình trạng

# Định dạng phản hồi
Chỉ cung cấp đối tượng JSON hợp lệ với cấu trúc này:
{
  "results": [
    {"result": "văn bản trích xuất hoặc null"},
    {"result": "văn bản trích xuất hoặc null"},
    ...
  ]
}

# Ví dụ đầu vào/đầu ra
Đầu vào:
{
  "records": [
    {"reason_combined": "Đau đầu, chóng mặt, buồn nôn", "specialist_name": "Thần kinh"},
    {"reason_combined": "Xuất huyết dạ dày, đau bung", "specialist_name": "Tiêu hóa"},
    {"reason_combined": "Khám tổng quát", "specialist_name": "Nội khoa"},
    {"reason_combined": "Đã mổ ruột thừa 2 năm trước", "specialist_name": "Ngoại khoa"}
  ]
https://nam-m9jrkzng-swedencentral.openai.azure.com}

Đầu ra:
{
  "results": [
    {"result": "đau đầu, chóng mặt, buồn nôn"},
    {"result": "xuất huyết dạ dày, đau bụng"},
    {"result": "khám tổng quát"},
    {"result": null}
  ]
}
"""


client = Together(
    api_key="95981b63ee8bc37a47e013a2946aecab679285179eb3a3c24e1179e38e75c434"
)
model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

def batch_process_records(dataframe, batch_size=100, output_path='/home/nampq/projects/specialist-prediction/data/data_version3/dataset_processed.csv'):
    """
    Process the dataframe in batches and update with results after each batch.
    Skips already processed records if the output file exists.
    """
    # Initialize or load processed dataframe
    if os.path.exists(output_path):
        processed_df = pd.read_csv(output_path)
        print(f"Found existing processed file with {len(processed_df)} records")
        
        # Check if we need to expand the processed dataframe (if input is larger)
        if len(dataframe) > len(processed_df):
            print(f"Input dataframe has {len(dataframe)} records, adding {len(dataframe) - len(processed_df)} new records")
            # Add new rows from dataframe that aren't in processed_df
            processed_df = pd.concat([processed_df, dataframe.iloc[len(processed_df):].copy()], ignore_index=False)
    else:
        processed_df = dataframe.copy()
        if 'processed_symptoms' not in processed_df.columns:
            processed_df['processed_symptoms'] = None
        print(f"Created new processed dataframe with {len(processed_df)} records")
    
    # Count records that need processing
    records_to_process = processed_df['processed_symptoms'].isna().sum()
    print(f"Found {records_to_process} records that need processing")
    
    if records_to_process == 0:
        print("All records are already processed. Nothing to do.")
        return processed_df
    
    total_records = len(processed_df)
    
    # Process only unprocessed records (where processed_symptoms is None/NaN)
    unprocessed_indices = processed_df[processed_df['processed_symptoms'].isna()].index.tolist()
    
    for batch_start in tqdm(range(0, len(unprocessed_indices), batch_size), desc="Processing data..."):
        batch_indices = unprocessed_indices[batch_start:batch_start + batch_size]
        
        print(f"Processing batch {batch_start//batch_size + 1}, records {batch_indices[0]} to {batch_indices[-1]}")
        
        # Extract batch records using the indices
        batch_df = processed_df.loc[batch_indices].copy()
        
        # Create input structure for the batch
        batch_records = []
        for _, row in batch_df.iterrows():
            batch_records.append({
                "reason_combined": row.get('reason_combind', ''),
                "specialist_name": row.get('specialist_name', '')
            })
        
        input_json = {"records": batch_records}
        print(f"Processing {len(batch_records)} records")
        
        # Process batch with Together API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Input: " + json.dumps(input_json)}
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        
        # Parse the response
        try:
            results_text = response.choices[0].message.content
            # Handle both JSON and string responses
            try:
                results_data = json.loads(results_text)
                print(results_data)
                if "results" in results_data:
                    batch_results = results_data["results"]
                    # Update the processed_symptoms column for this batch
                    for i, result in enumerate(batch_results):
                        if i < len(batch_indices):
                            index = batch_indices[i]
                            processed_df.at[index, 'processed_symptoms'] = result.get('result')
                else:
                    print(f"Warning: Unexpected response format in batch {batch_start//batch_size + 1}")
            except json.JSONDecodeError:
                print(f"Warning: Response is not valid JSON in batch {batch_start//batch_size + 1}")
                lines = results_text.strip().split('\n')
                for i, line in enumerate(lines):
                    if i < len(batch_indices):
                        index = batch_indices[i]
                        processed_df.at[index, 'processed_symptoms'] = line if line.strip() else None
        except Exception as e:
            print(f"Error processing response in batch {batch_start//batch_size + 1}: {str(e)}")
        
        # Save the dataframe after each batch
        processed_df.to_csv(output_path, index=False)
        print(f"Progress saved after batch {batch_start//batch_size + 1}")
    
    print(f"Processing complete. Processed {records_to_process} out of {total_records} records.")
    return processed_df

# Process the dataset in batches
batch_size = 10
processed_df = batch_process_records(df, batch_size)

# Examine a few results
print("\nSample results:")
for i in range(min(5, len(processed_df))):
    print(f"Record {i+1}:")
    print(f"Reason: {processed_df.iloc[i]['reason_combind']}")
    print(f"Specialist: {processed_df.iloc[i]['specialist_name']}")
    print(f"Processed Symptoms: {processed_df.iloc[i]['processed_symptoms']}")
    print()