use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

/// 單一網格資料 (純 Vec<Vec<i32>>)
pub type GridData = Vec<Vec<i32>>;

/// 單一 Input/Output 對
/// ⚠️ output 使用 Option 以處理 Kaggle 隱藏測試集 (output 欄位缺失)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub input: GridData,
    pub output: Option<GridData>,
}

impl Example {
    /// 檢查是否有訓練標籤
    pub fn has_label(&self) -> bool {
        self.output.is_some()
    }

    /// 取得 output (若有)
    pub fn get_output(&self) -> Option<&GridData> {
        self.output.as_ref()
    }
}

/// 單一任務 (包含訓練集與測試集)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub train: Vec<Example>,
    pub test: Vec<Example>,
}

/// 整個 ARC 資料集 (Task ID -> Task)
pub type ArcDataset = HashMap<String, Task>;

/// 從檔案載入 ARC 資料集
pub fn load_dataset(file_path: &str) -> Result<ArcDataset, Box<dyn std::error::Error>> {
    let data = fs::read_to_string(file_path)?;
    let dataset: ArcDataset = serde_json::from_str(&data)?;
    Ok(dataset)
}

/// 將結果序列化為 Kaggle 要求的 submission JSON
pub fn save_submissions(submissions: &HashMap<String, GridData>, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(submissions)?;
    let tmp_path = format!("{}.tmp", output_path);
    fs::write(&tmp_path, &json)?;
    fs::rename(&tmp_path, output_path)?;
    Ok(())
}

/// 從 GridData 轉換為 Grid 結構 (用於後續的 Genetic Engine)
pub fn grid_data_to_grid(data: &GridData) -> crate::Grid {
    crate::Grid::from_vec(data.clone())
}

/// 從 Grid 結構轉換為 GridData (用於輸出) - converts flat Vec<i32> back to nested format
pub fn grid_to_grid_data(grid: &crate::Grid) -> GridData {
    let mut nested = Vec::with_capacity(grid.height);
    for r in 0..grid.height {
        let row_start = r * grid.width;
        nested.push(grid.data[row_start..row_start + grid.width].to_vec());
    }
    nested
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_sample_json(path: &str, include_test_output: bool) {
        let json = if include_test_output {
            r#"{
  "task_0": {
    "train": [
      {
        "input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
        "output": [[0, 0, 0], [0, 3, 0], [0, 0, 0]]
      }
    ],
    "test": [
      {
        "input": [[2, 2], [2, 2]],
        "output": [[3, 3], [3, 3]]
      }
    ]
  },
  "task_1": {
    "train": [
      {
        "input": [[1, 1], [1, 1]],
        "output": [[2, 2], [2, 2]]
      }
    ],
    "test": [
      {
        "input": [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
      }
    ]
  }
}"#
        } else {
            r#"{
  "task_0": {
    "train": [
      {
        "input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
        "output": [[0, 0, 0], [0, 3, 0], [0, 0, 0]]
      }
    ],
    "test": [
      {
        "input": [[2, 2], [2, 2]]
      }
    ]
  }
}"#
        };

        let mut file = fs::File::create(path).expect("無法建立測試檔案");
        file.write_all(json.as_bytes()).expect("無法寫入測試檔案");
    }

    #[test]
    fn test_load_dataset_with_test_output() {
        let temp_file = "/tmp/fiuld_test_with_output.json";
        create_sample_json(temp_file, true);

        let dataset = load_dataset(temp_file).expect("無法載入資料集");

        // 檢查有 2 個任務
        assert_eq!(dataset.len(), 2);

        // 檢查 task_0
        let task_0 = dataset.get("task_0").expect("找不到 task_0");
        assert_eq!(task_0.train.len(), 1);
        assert_eq!(task_0.test.len(), 1);
        
        // 檢查訓練資料的 output
        assert!(task_0.train[0].has_label());
        let train_output = task_0.train[0].get_output().unwrap();
        assert_eq!(train_output[1][1], 3);

        // 檢查測試資料的 output
        assert!(task_0.test[0].has_label());
        let test_output = task_0.test[0].get_output().unwrap();
        assert_eq!(test_output[0][0], 3);

        // 檢查 task_1
        let task_1 = dataset.get("task_1").expect("找不到 task_1");
        assert_eq!(task_1.test[0].input.len(), 3);
        assert_eq!(task_1.test[0].input[0].len(), 3);
    }

    #[test]
    fn test_load_dataset_without_test_output() {
        let temp_file = "/tmp/fiuld_test_no_output.json";
        create_sample_json(temp_file, false);

        let dataset = load_dataset(temp_file).expect("無法載入資料集");

        // 檢查 task_0
        let task_0 = dataset.get("task_0").expect("找不到 task_0");
        
        // 測試資料沒有 output
        assert!(!task_0.test[0].has_label());
        assert!(task_0.test[0].get_output().is_none());

        // 但訓練資料有 output
        assert!(task_0.train[0].has_label());

        // 測試資料的 input 仍然正確
        assert_eq!(task_0.test[0].input.len(), 2);
        assert_eq!(task_0.test[0].input[0][0], 2);
    }

    #[test]
    fn test_save_submissions() {
        let mut submissions = HashMap::new();
        submissions.insert("task_0_test_0".to_string(), vec![vec![1, 2], vec![3, 4]]);
        submissions.insert("task_1_test_0".to_string(), vec![vec![5, 5, 5]]);

        let temp_file = "/tmp/fiuld_submission.json";
        save_submissions(&submissions, temp_file).expect("無法儲存 submission");

        // 驗證檔案存在且內容正確
        let data = fs::read_to_string(temp_file).expect("無法讀取 submission 檔案");
        println!("Saved JSON:\n{}", data);
        assert!(data.contains("task_0_test_0"));
        assert!(data.contains("task_1_test_0"));
        assert!(data.contains("1") && data.contains("2") && data.contains("3") && data.contains("4"));
    }

    #[test]
    fn test_grid_data_to_grid() {
        let data = vec![vec![0, 1, 0], vec![1, 2, 1], vec![0, 1, 0]];
        let grid = grid_data_to_grid(&data);

        assert_eq!(grid.height, 3);
        assert_eq!(grid.width, 3);
        assert_eq!(grid.get(1, 1), 2);
        assert_eq!(grid.get(0, 1), 1);
    }

    #[test]
    fn test_grid_to_grid_data() {
        let mut grid = crate::Grid::new(2, 3, 0);
        grid.set(0, 0, 1);
        grid.set(1, 2, 2);

        let data = grid_to_grid_data(&grid);
        assert_eq!(data[0][0], 1);
        assert_eq!(data[1][2], 2);
        assert_eq!(data[0][1], 0);
    }

    #[test]
    fn test_example_methods() {
        let example_with_output = Example {
            input: vec![vec![1, 2]],
            output: Some(vec![vec![3, 4]]),
        };
        assert!(example_with_output.has_label());
        assert!(example_with_output.get_output().is_some());
        assert_eq!(example_with_output.get_output().unwrap()[0][0], 3);

        let example_without_output = Example {
            input: vec![vec![1, 2]],
            output: None,
        };
        assert!(!example_without_output.has_label());
        assert!(example_without_output.get_output().is_none());
    }
}
