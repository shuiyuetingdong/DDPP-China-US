import dspy
from typing import List, Dict, Tuple, Optional, Literal, Type
import pandas as pd
from tqdm import tqdm
import time
import traceback
import os
from datetime import datetime
import shutil
import yaml
import dspy_signature
import inspect

class APIConfigManager:

    def __init__(self, config_path: str = "config.yaml"):

        self.config = self._load_config(config_path)
        self._initialize_llm()
    
    def _load_config(self, config_path: str) -> dict:

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if 'api' not in config:
                    raise ValueError("Missing 'api' configuration in config file")
                return config['api']
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Config file format error: {e}")
    
    def _initialize_llm(self):
        gpt_4o = dspy.LM(
            self.config['model'],
            api_key=self.config['key'],
            api_base=self.config['base'],
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens'],
            stop=None,
            cache=False
        )
        dspy.configure(lm=gpt_4o)
    
    def reload_config(self, config_path: str):

        self.config = self._load_config(config_path)
        self._initialize_llm()
        print(f"Config file reloaded: {config_path}")

api_manager = APIConfigManager()


class SignatureManager:

    @staticmethod
    def extract_fields_from_signature(signature_class) -> Dict[str, Dict[str, str]]:
        """
        Extract field information from dspy_signature class
        
        Args:
            signature_class: DSPy signature class
            
        Returns:
            dict: Dictionary containing field information
        """
        available_fields = list(signature_class.__annotations__.keys())
        
        return {
            'available_fields': available_fields
        }
    
    @staticmethod
    def get_task_class(task_type: str) -> Type[dspy.Signature]:

        try:
            return getattr(dspy_signature, task_type)
        except AttributeError:
            raise ValueError(f"Task class not found in dspy_signature module: {task_type}")
    
    @staticmethod
    def get_available_tasks() -> List[str]:

        return [
            name for name, cls in inspect.getmembers(dspy_signature, inspect.isclass)
            if issubclass(cls, dspy.Signature) and cls != dspy.Signature
        ]


def create_backup(file_path: str, backup_dir: str) -> str:

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.basename(file_path)
    backup_name = f"{os.path.splitext(file_name)[0]}_{timestamp}.csv"
    backup_path = os.path.join(backup_dir, backup_name)
    
    shutil.copy2(file_path, backup_path)
    print(f"Backup file created: {backup_path}")
    return backup_path

def get_last_processed_index(df: pd.DataFrame, result_column: str = "toxic_result") -> int:

    if result_column not in df.columns:
        return 0
    
    processed_mask = df[result_column].notna()
    if not processed_mask.any():
        return 0
    
    return df[processed_mask].index[-1] + 1

def process_comments_batch(
    input_file_path: str,
    task_type: str,
    output_dir: str = "results",
    backup_dir: str = "backups",
    batch_size: int = 10,
    start_index: int = None,
    input_mapping: Dict[str, str] = None,  # Optional input field mapping override
    output_mapping: Dict[str, str] = None,  # Optional output field mapping override
    max_rows: Optional[int] = None,  # Control maximum number of rows to process, None means process all rows
    config_path: str = None,  # Config file path, if None use default config
    max_retries: int = 3  # New parameter: maximum retry attempts
):

    '''
    Args:
    input_file_path: Enter the path of the CSV file
    task_type: task type
    output_dir: indicates the directory for storing the result file
    backup_dir: backup file storage directory
    batch_size: The number of comments per batch
    start_index: The index position at which the processing began. If None, it automatically continues from the last processing position
    input_mapping: Optional input field mapping that overrides the default field mapping
    output_mapping: Optional output field mapping that overrides the default field mapping
    max_rows: The maximum number of rows to be processed, or all rows if None
    config_path: configuration file path, if None, the default configuration will be used
    max_retries: indicates the maximum number of retries when an LLM call fails
    '''

    if config_path is not None:
        api_manager.reload_config(config_path)
    
    print(SignatureManager.get_available_tasks())

    task_class = SignatureManager.get_task_class(task_type)
    fields_info = SignatureManager.extract_fields_from_signature(task_class)
    
    task_instance = dspy.Predict(task_class)
    
    if input_mapping is None or output_mapping is None:
        raise ValueError(f"Please provide field mappings (input_mapping and output_mapping).\n"
                         f"Available fields are: {fields_info['available_fields']}, please refer to dspy_signature.py documentation")

    invalid_input_fields = [field for field in input_mapping.keys() if field not in fields_info['available_fields']]
    invalid_output_fields = [field for field in output_mapping.keys() if field not in fields_info['available_fields']]
    
    if invalid_input_fields or invalid_output_fields:
        error_msg = "Field mapping error:\n"
        if invalid_input_fields:
            error_msg += f"Fields in input mapping {invalid_input_fields} do not exist\n"
        if invalid_output_fields:
            error_msg += f"Fields in output mapping {invalid_output_fields} do not exist\n"
        error_msg += f"Available fields are: {fields_info['available_fields']}, please refer to dspy_signature.py documentation"
        raise ValueError(error_msg)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_filename = os.path.basename(input_file_path)
    output_filename = f"processed_{os.path.splitext(input_filename)[0]}.csv"
    output_file_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_file_path):
        df = pd.read_csv(output_file_path)
        print(f"Continuing processing existing result file: {output_file_path}")
    else:
        df = pd.read_csv(input_file_path)
        for output_name in output_mapping.keys():
            column_name = output_mapping[output_name]
            if column_name not in df.columns:
                df[column_name] = None
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"Created new result file: {output_file_path}")
    
    missing_columns = [col for col in input_mapping.values() if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing in input file: {missing_columns}")
    
    total_rows = len(df)
    

    if start_index is None:
        first_output_column = next(iter(output_mapping.values()))
        start_index = get_last_processed_index(df, first_output_column)
        print(f"Continuing from last processing position: record {start_index}")
    
    if max_rows is not None:
        total_rows = min(start_index + max_rows, total_rows)
        print(f"Will process {max_rows} records starting from record {start_index}")
    
    backup_path = create_backup(output_file_path, backup_dir)
    
    with tqdm(total=total_rows-start_index) as pbar:
        for i in range(start_index, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            
            try:
                for idx in range(i, batch_end):
                    first_output_column = next(iter(output_mapping.values()))
                    if pd.notna(df.loc[idx, first_output_column]):
                        pbar.update(1)
                        continue
                    
                    inputs = {
                        llm_field: str(df.iloc[idx][csv_field])
                        for llm_field, csv_field in input_mapping.items()
                    }
                    
                    retry_count = 0
                    success = False
                    last_error = None
                    
                    while retry_count <= max_retries:  # 最多尝试 max_retries+1 次（1次初始 + max_retries次重试）
                        try:
                            result = task_instance(**inputs)
                            
                            for llm_field, csv_field in output_mapping.items():
                                df.loc[idx, csv_field] = result.get(llm_field, None)
                            
                            success = True
                            break
                            
                        except Exception as e:
                            last_error = e
                            if retry_count < max_retries:
                                retry_count += 1
                                print(f"\nError occurred while processing record {idx}, attempting retry {retry_count}:")
                                print(str(e))
                                time.sleep(1)
                            else:
                                print(f"\nFailed to process record {idx} after {max_retries} retries, skipping this record:")
                                print(str(last_error))
                                for llm_field, csv_field in output_mapping.items():
                                    df.loc[idx, csv_field] = f"Processing failed: {str(last_error)}"
                                break
                    
                    pbar.update(1) 
                
                df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
                print(f"\nSaved batch {i//batch_size + 1} processing results (processed up to record {batch_end})")
                
            except Exception as e:
                print(f"\nError occurred while processing records {i} to {batch_end}:")
                print(traceback.format_exc())
                print(f"Current progress saved, you can continue from index {i}")
                df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    
    print(f"Processing complete! Results saved to: {output_file_path}")
    print(f"Backup file location: {backup_path}")
    return df


if __name__ == "__main__":
    try:
        df_result = process_comments_batch(
            start_index=None, 
            input_file_path="processed_single_unfitered.csv", # Input file path
            task_type="USCN_comment_detection_CN2US_dspy", # Task type
            output_dir="results_gpt4o", # Result file storage directory
            backup_dir="backups_gpt4o", # Backup file storage directory
            batch_size=10, # Number of comments per batch

            input_mapping={
                "短视频题目": "label_3",
                "评论": "comment",  # 输入映射，'评论'是LLM输入字段名，content是CSV文件中的用作输入的列名
                "ip地址": "ip_label"
            },
            output_mapping={    
                "情感分类": "answer_dspy",  # 输出映射，'分析结果'是LLM输出字段名
            },

            config_path="config.yaml",
            max_retries=2
        )
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except yaml.YAMLError as e:
        print(f"Config file format error: {e}")
    except ValueError as e:
        print(f"Parameter error: {e}")
    except Exception as e:
        print(f"Unexpected error occurred:")
        print(traceback.format_exc())
    finally:
        print("\nLast 3 LLM calls:")
        dspy.inspect_history(n=3)


