use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;
use rayon::prelude::*;

// Global debug log file
lazy_static::lazy_static! {
    static ref DEBUG_LOG: Mutex<Option<File>> = Mutex::new(None);
}

fn init_debug_log() -> Result<(), Box<dyn std::error::Error>> {
    let log_path = "/kaggle/working/fiuld_debug.log";
    let file = File::create(log_path)?;
    let mut log = DEBUG_LOG.lock().unwrap();
    *log = Some(file);
    Ok(())
}

fn debug_log(msg: &str) {
    if let Ok(mut log) = DEBUG_LOG.lock() {
        if let Some(ref mut file) = *log {
            let _ = writeln!(file, "{} {}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0), msg);
            let _ = file.flush();
        }
    }
}

mod data;
mod genetic;
mod reflection;

// ============================================================================
// 核心資料結構：Grid (Flat layout for cache-friendly & SIMD access)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid {
    pub data: Vec<i32>,  // Flat: data[r * width + c]
    pub height: usize,
    pub width: usize,
}

impl Grid {
    #[inline]
    fn idx(r: usize, c: usize, width: usize) -> usize {
        r * width + c
    }

    pub fn new(height: usize, width: usize, fill_value: i32) -> Self {
        Grid {
            data: vec![fill_value; height * width],
            height,
            width,
        }
    }

    pub fn from_vec(vec: Vec<Vec<i32>>) -> Self {
        let height = vec.len();
        let width = if height > 0 { vec[0].len() } else { 0 };
        let flat: Vec<i32> = vec.into_iter().flatten().collect();
        Grid { data: flat, height, width }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> i32 {
        if r < self.height && c < self.width {
            self.data[Self::idx(r, c, self.width)]
        } else {
            0
        }
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, value: i32) {
        if r < self.height && c < self.width {
            self.data[Self::idx(r, c, self.width)] = value;
        }
    }

    /// 計算兩個網格的 exact match ratio (flat layout = SIMD friendly)
    pub fn match_ratio(&self, other: &Grid) -> f64 {
        if self.height != other.height || self.width != other.width {
            return 0.0;
        }
        let total = (self.height * self.width) as f64;
        let matches = self.data.iter().zip(&other.data)
            .filter(|&(a, b)| a == b)
            .count() as f64;
        matches / total
    }

    pub fn clone(&self) -> Grid {
        Grid { data: self.data.clone(), height: self.height, width: self.width }
    }

    pub fn print(&self) {
        for r in 0..self.height {
            let row: Vec<String> = (0..self.width)
                .map(|c| format!("{:2}", self.get(r, c)))
                .collect();
            println!("{}", row.join(" "));
        }
    }

    /// 將 grid 轉回 Vec<Vec<i32>> (用於輸出)
    pub fn to_vec(&self) -> Vec<Vec<i32>> {
        (0..self.height)
            .map(|r| self.data[r * self.width .. (r + 1) * self.width].to_vec())
            .collect()
    }
}

// ============================================================================
// 物件資料結構：Object (連通分量)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    pub color: i32,
    pub pixels: Vec<(usize, usize)>, // (row, col)
    pub centroid: (f64, f64),        // 質心 (用於移動目標計算)
}

impl Object {
    pub fn new(color: i32, pixels: Vec<(usize, usize)>) -> Self {
        let n = pixels.len() as f64;
        let cy = pixels.iter().map(|(r, _c)| *r as f64).sum::<f64>() / n;
        let cx = pixels.iter().map(|(_r, c)| *c as f64).sum::<f64>() / n;
        Object {
            color,
            pixels,
            centroid: (cy, cx),
        }
    }

    /// 將物件繪製到網格
    pub fn draw_to_grid(&self, grid: &mut Grid) {
        for &(r, c) in &self.pixels {
            grid.set(r, c, self.color);
        }
    }
}

// ============================================================================
// 從網格提取物件 (連通分量分析)
// ============================================================================

pub fn grid_to_objects(grid: &Grid) -> Vec<Object> {
    let mut visited = vec![vec![false; grid.width]; grid.height];
    let mut objects = Vec::new();

    // 4-way connectivity (上下左右)
    let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];

    for r in 0..grid.height {
        for c in 0..grid.width {
            if !visited[r][c] && grid.get(r, c) != 0 {
                let color = grid.get(r, c);
                let mut pixels = Vec::new();
                let mut stack = vec![(r, c)];

                while let Some((cr, cc)) = stack.pop() {
                    if visited[cr][cc] {
                        continue;
                    }
                    visited[cr][cc] = true;
                    pixels.push((cr, cc));

                    for &(dr, dc) in &directions {
                        let nr = cr as i32 + dr;
                        let nc = cc as i32 + dc;
                        if nr >= 0 && nr < grid.height as i32 && nc >= 0 && nc < grid.width as i32 {
                            let (nr, nc) = (nr as usize, nc as usize);
                            if !visited[nr][nc] && grid.get(nr, nc) == color {
                                stack.push((nr, nc));
                            }
                        }
                    }
                }

                if !pixels.is_empty() {
                    objects.push(Object::new(color, pixels));
                }
            }
        }
    }

    objects
}

// ============================================================================
// 從物件重建網格
// ============================================================================

pub fn objects_to_grid(objects: &[Object], height: usize, width: usize) -> Grid {
    let mut grid = Grid::new(height, width, 0);
    for obj in objects {
        obj.draw_to_grid(&mut grid);
    }
    grid
}

// ============================================================================
// Kaggle 任務結構
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub training: Vec<TestCase>,
    pub test: Vec<TestCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub input: Grid,
    pub output: Grid,
}

// ============================================================================
// 求解器核心 (框架)
// ============================================================================

pub struct Solver {
    max_time_seconds: f64,
    config: genetic::EvolutionConfig,
}

impl Solver {
    pub fn new(max_time_seconds: f64) -> Self {
        Solver {
            max_time_seconds,
            config: genetic::EvolutionConfig {
                population_size: 2000,
                generations: 150,
                mutation_rate: 0.15,
                elite_ratio: 0.1,
                max_actions: 10,
                target_fitness: 0.999,
                patience: 80,
            },
        }
    }

    /// 單一任務求解 (整合 Genetic Engine)
    pub fn solve(&self, example: &data::Example, train_examples: &[data::Example]) -> Grid {
        let input = &example.input;
        let target = example.get_output();

        // 轉換為 Grid 結構
        let input_grid = Grid::from_vec(input.clone());
        
        println!("=== Solving task ===");
        println!("Input: {}x{}", input_grid.height, input_grid.width);

        // 建立訓練對
        let train_pairs: Vec<(Grid, Grid)> = train_examples
            .iter()
            .filter_map(|ex| {
                ex.get_output().map(|target_data| {
                    (Grid::from_vec(ex.input.clone()), Grid::from_vec(target_data.clone()))
                })
            })
            .collect();

        if train_pairs.is_empty() {
            println!("⚠️ No training data available, returning input as fallback");
            return input_grid;
        }

        println!("Training pairs: {}", train_pairs.len());

        if let Some(target_data) = target {
            let target_grid = Grid::from_vec(target_data.clone());
            println!("Target: {}x{}", target_grid.height, target_grid.width);
        } else {
            println!("Target: (hidden - Kaggle test case)");
        }

        // 使用 Genetic Engine 求解
        let start = std::time::Instant::now();
        let mut result = genetic::evolve(&train_pairs, input_grid.height, input_grid.width, &self.config);
        
        // 如果沒達到目標，啟動 Reflection Loop
        let mut reflection_count = 0;
        while result.best_fitness < self.config.target_fitness {
            // 分析最佳解的錯誤
            let predictions: Vec<(Grid, Grid)> = train_pairs
                .iter()
                .map(|(input, target)| {
                    let predicted = result.best_genome.execute(input);
                    (predicted, target.clone())
                })
                .collect();

            let report = reflection::ReflectionEngine::analyze_all(&predictions);
            
            if !reflection::ReflectionEngine::should_reflect(&report, self.config.target_fitness) {
                break;
            }

            println!("  🔍 Reflection #{}: {} errors, avg_match={:.3}", 
                reflection_count + 1, report.total_errors, report.avg_match_ratio);

            // 生成補丁
            let patches = reflection::ReflectionEngine::generate_patches(&report);
            
            if patches.is_empty() {
                break;
            }

            // 印出補丁細節
            for (i, patch) in patches.iter().enumerate() {
                println!("  📝 Patch {}: {:?}", i + 1, patch);
            }

            // 應用補丁到最佳基因體
            let mut patched_genome = result.best_genome.clone();
            reflection::ReflectionEngine::apply_patches(&mut patched_genome, &patches);

            // 驗證補丁效果
            let patched_fitness = patched_genome.evaluate_batch(&train_pairs);

            println!("  🧬 Injected {} patch(es), patched_fitness={:.4}", 
                patches.len(), patched_fitness);

            if patched_fitness <= result.best_fitness {
                println!("  ⚠️ Patches didn't help, skipping reflection");
                break;
            }

            // 以補丁基因體為種子，重新演化
            let mut rng = rand::thread_rng();
            let mut next_population: Vec<genetic::Genome> = vec![patched_genome];
            
            while next_population.len() < self.config.population_size {
                if next_population.len() % 2 == 0 && next_population.len() > 1 {
                    let parent1 = &next_population[next_population.len() - 2];
                    let parent2 = &next_population[next_population.len() - 1];
                    let child = parent1.crossover(parent2, &mut rng);
                    let mut child = child;
                    child.mutate(&mut rng, self.config.mutation_rate, input_grid.height, input_grid.width);
                    next_population.push(child);
                } else {
                    next_population.push(genetic::Genome::new_random(&mut rng, input_grid.height, input_grid.width, self.config.max_actions));
                }
            }

            // 平行評估
            let mut scored: Vec<(genetic::Genome, f64)> = next_population
                .into_par_iter()
                .map(|g| {
                    let fitness = g.evaluate_batch(&train_pairs);
                    (g, fitness)
                })
                .collect();

            let (best_idx, best_fitness_val) = scored
                .iter()
                .enumerate()
                .max_by(|(_, (_, f1)), (_, (_, f2))| f1.partial_cmp(f2).unwrap())
                .map(|(i, (_, f))| (i, *f))
                .unwrap();

            let (best_genome, _) = scored.remove(best_idx);

            result.best_genome = best_genome;
            result.best_fitness = best_fitness_val;

            result.generations_run += 1;
            reflection_count += 1;

            if reflection_count >= 3 {
                println!("  ⚠️ Max reflections reached");
                break;
            }
        }

        let elapsed = start.elapsed();

        println!("  🏆 Best fitness: {:.4}", result.best_fitness);
        println!("  🧬 Actions: {}", result.best_genome.actions.len());
        println!("  🔄 Reflections: {}", reflection_count);
        println!("  ⏱️  Time: {:?}", elapsed);

        let solution = result.best_genome.execute(&input_grid);
        solution
    }

    /// 批量求解 (使用 Rayon 平行化)
    pub fn solve_batch(&self, tasks: &[data::Task]) -> Vec<HashMap<String, Grid>> {
        // 使用 rayon 平行處理每個任務
        tasks.par_iter().map(|task| {
            let mut results = HashMap::new();

            for (i, example) in task.test.iter().enumerate() {
                let solution = self.solve(example, &task.train);
                let key = format!("test_{}", i);
                results.insert(key, solution);
            }

            results
        }).collect()
    }
}

// ============================================================================
// DSL 動作編譯器 (從 V28 移植)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DslAction {
    Fill { color: i32, x: usize, y: usize, w: usize, h: usize },
    DrawLine { color: i32, x1: usize, y1: usize, x2: usize, y2: usize },
    DrawRect { color: i32, x: usize, y: usize, w: usize, h: usize },
    MoveObject { from_x: usize, from_y: usize, to_x: usize, to_y: usize },
    Rotate90,
    Rotate180,
    Rotate270,
    FlipHorizontal,
    FlipVertical,
    ColorSwap { from_color: i32, to_color: i32 },
    Scale { factor: usize },
    Crop { x: usize, y: usize, w: usize, h: usize },
    Pad { top: usize, bottom: usize, left: usize, right: usize, color: i32 },
    Clear,
    Copy,
}

impl DslAction {
    #[inline]
    fn idx(r: usize, c: usize, width: usize) -> usize {
        r * width + c
    }

    /// 將 DSL 動作套用至網格 (Flat layout optimized)
    #[inline]
    pub fn apply(&self, grid: &mut Grid) -> bool {
        match self {
            DslAction::Fill { color, x, y, w, h } => {
                let end_r = (*y + *h).min(grid.height);
                let end_c = (*x + *w).min(grid.width);
                for r in *y..end_r {
                    let row_start = Self::idx(r, 0, grid.width);
                    for c in *x..end_c {
                        let idx = row_start + c;
                        grid.data[idx] = *color;
                    }
                }
                true
            }
            DslAction::DrawLine { color, x1, y1, x2, y2 } => {
                let dx = (*x2 as i32 - *x1 as i32).abs();
                let dy = (*y2 as i32 - *y1 as i32).abs();
                let sx = if *x1 <= *x2 { 1 } else { -1 };
                let sy = if *y1 <= *y2 { 1 } else { -1 };
                let mut err = dx as i32 - dy as i32;
                let mut cx = *x1 as i32;
                let mut cy = *y1 as i32;

                while (cx != *x2 as i32 || cy != *y2 as i32) && cx >= 0 && cx < grid.width as i32 && cy >= 0 && cy < grid.height as i32 {
                    if cy < (grid.height) as i32 && cx < grid.width as i32 {
                        let idx = Self::idx(cy as usize, cx as usize, grid.width);
                        grid.data[idx] = *color;
                    }
                    let e2 = 2 * err;
                    if e2 > -dy as i32 { err -= dy as i32; cx += sx; }
                    if e2 < dx as i32 { err += dx as i32; cy += sy; }
                }
                true
            }
            DslAction::DrawRect { color, x, y, w, h } => {
                let end_r = (*y + *h).min(grid.height);
                let end_c = (*x + *w).min(grid.width);
                for r in *y..end_r {
                    let row_start = Self::idx(r, 0, grid.width);
                    for c in *x..end_c {
                        if r == *y || r == (*y + *h - 1) || c == *x || c == (*x + *w - 1) {
                            grid.data[row_start + c] = *color;
                        }
                    }
                }
                true
            }
            DslAction::ColorSwap { from_color, to_color } => {
                // Hot path: direct flat slice iteration (SIMD-friendly)
                for pixel in grid.data.iter_mut() {
                    if *pixel == *from_color {
                        *pixel = *to_color;
                    }
                }
                true
            }
            DslAction::Clear => {
                grid.data.iter_mut().for_each(|p| *p = 0);
                true
            }
            DslAction::Rotate90 => {
                let new_h = grid.width;
                let new_w = grid.height;
                let mut new_data = vec![0i32; new_h * new_w];
                for r in 0..grid.height {
                    let row_start = Self::idx(r, 0, grid.width);
                    for c in 0..grid.width {
                        let src = row_start + c;
                        new_data[Self::idx(c, new_h - 1 - r, new_w)] = grid.data[src];
                    }
                }
                grid.data = new_data;
                grid.height = new_h;
                grid.width = new_w;
                true
            }
            DslAction::Rotate180 => {
                let mut new_data = vec![0i32; grid.height * grid.width];
                for r in 0..grid.height {
                    let row_start = Self::idx(r, 0, grid.width);
                    for c in 0..grid.width {
                        let src = row_start + c;
                        new_data[Self::idx(grid.height - 1 - r, grid.width - 1 - c, grid.width)] =
                            grid.data[src];
                    }
                }
                grid.data = new_data;
                true
            }
            DslAction::Rotate270 => {
                let new_h = grid.width;
                let new_w = grid.height;
                let mut new_data = vec![0i32; new_h * new_w];
                for r in 0..grid.height {
                    let row_start = Self::idx(r, 0, grid.width);
                    for c in 0..grid.width {
                        let src = row_start + c;
                        new_data[Self::idx(new_w - 1 - c, r, new_w)] = grid.data[src];
                    }
                }
                grid.data = new_data;
                grid.height = new_h;
                grid.width = new_w;
                true
            }
            DslAction::FlipHorizontal => {
                let mut new_data = vec![0i32; grid.height * grid.width];
                for r in 0..grid.height {
                    let row_start = Self::idx(r, 0, grid.width);
                    for c in 0..grid.width {
                        new_data[row_start + (grid.width - 1 - c)] = grid.data[row_start + c];
                    }
                }
                grid.data = new_data;
                true
            }
            DslAction::FlipVertical => {
                let mut new_data = vec![0i32; grid.height * grid.width];
                for r in 0..grid.height {
                    let src_start = Self::idx(r, 0, grid.width);
                    let dst_start = Self::idx(grid.height - 1 - r, 0, grid.width);
                    new_data[dst_start..dst_start + grid.width].copy_from_slice(&grid.data[src_start..src_start + grid.width]);
                }
                grid.data = new_data;
                true
            }
            DslAction::Scale { factor } => {
                let f = *factor as usize;
                if f == 0 || grid.data.is_empty() {
                    return true;
                }
                let new_h = grid.height * f;
                let new_w = grid.width * f;
                let mut new_data = vec![0i32; new_h * new_w];
                for r in 0..grid.height {
                    let row_start = Self::idx(r, 0, grid.width);
                    for c in 0..grid.width {
                        let color = grid.data[row_start + c];
                        for dr in 0..f {
                            for dc in 0..f {
                                let nr = r * f + dr;
                                let nc = c * f + dc;
                                new_data[Self::idx(nr, nc, new_w)] = color;
                            }
                        }
                    }
                }
                grid.data = new_data;
                grid.height = new_h;
                grid.width = new_w;
                true
            }
            DslAction::Crop { x, y, w, h } => {
                let end_r = (*y + *h).min(grid.height);
                let end_c = (*x + *w).min(grid.width);
                let new_h = end_r - *y;
                let new_w = end_c - *x;
                if new_h <= 0 || new_w <= 0 {
                    return true;
                }
                let mut new_data = vec![0i32; new_h * new_w];
                for r in 0..new_h {
                    let src_row = *y + r;
                    let src_start = Self::idx(src_row, *x, grid.width);
                    new_data[Self::idx(r, 0, new_w)..Self::idx(r, new_w, new_w)]
                        .copy_from_slice(&grid.data[src_start..src_start + new_w]);
                }
                grid.data = new_data;
                grid.height = new_h;
                grid.width = new_w;
                true
            }
            DslAction::Pad { top, bottom, left, right, color } => {
                let new_h = grid.height + top + bottom;
                let new_w = grid.width + left + right;
                let mut new_data = vec![*color; new_h * new_w];
                for r in 0..grid.height {
                    let src_start = Self::idx(r, 0, grid.width);
                    let dst_start = Self::idx(top + r, *left, new_w);
                    new_data[dst_start..dst_start + grid.width].copy_from_slice(&grid.data[src_start..src_start + grid.width]);
                }
                grid.data = new_data;
                grid.height = new_h;
                grid.width = new_w;
                true
            }
            DslAction::MoveObject { from_x, from_y, to_x, to_y } => {
                let dr = *to_y as i32 - *from_y as i32;
                let dc = *to_x as i32 - *from_x as i32;
                if dr == 0 && dc == 0 {
                    return true;
                }
                let mut new_data = vec![0i32; grid.height * grid.width];
                for r in 0..grid.height {
                    let row_start = Self::idx(r, 0, grid.width);
                    for c in 0..grid.width {
                        let color = grid.data[row_start + c];
                        if color == 0 { continue; }
                        let new_r = r as i32 + dr;
                        let new_c = c as i32 + dc;
                        if new_r >= 0 && new_r < grid.height as i32 && new_c >= 0 && new_c < grid.width as i32 {
                            let dst = Self::idx(new_r as usize, new_c as usize, grid.width);
                            new_data[dst] = color;
                        }
                    }
                }
                grid.data = new_data;
                true
            }
            DslAction::Copy => {
                // Copy 不做任何事，保留原網格 (用於基因組的分支)
                true
            }
        }
    }
}

// ============================================================================
// 主程式
// ============================================================================

fn main() {
    println!("🚀 Fiuld - ARC-AGI-3 Rust Agent");
    let start = Instant::now();

    // Initialize debug log
    if let Err(e) = init_debug_log() {
        eprintln!("⚠️ Failed to init debug log: {}", e);
    }
    debug_log("Fiuld engine starting");

    // 命令列參數
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <test.json> <output.json>", args[0]);
        eprintln!("Example: fiuld /kaggle/input/arc-prize-2026/test.json submission.json");
        std::process::exit(1);
    }

    let test_path = &args[1];
    let output_path = &args[2];

    println!("📂 Loading test data from: {}", test_path);
    debug_log(&format!("Loading test data from: {}", test_path));

    // 讀取並解析 JSON
    let dataset = match data::load_dataset(test_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("❌ Failed to load dataset: {}", e);
            debug_log(&format!("Failed to load dataset: {}", e));
            std::process::exit(1);
        }
    };

    println!("✅ Loaded {} tasks from {}", dataset.len(), test_path);
    debug_log(&format!("Loaded {} tasks", dataset.len()));

    // 統計訓練/測試資料
    let total_train = dataset.values().map(|t| t.train.len()).sum::<usize>();
    let total_test = dataset.values().map(|t| t.test.len()).sum::<usize>();
    println!("   Training examples: {}", total_train);
    println!("   Test cases: {}", total_test);
    debug_log(&format!("Training examples: {}, Test cases: {}", total_train, total_test));

    // 建立求解器 (9 小時限制)
    let solver = Solver::new(8.5 * 3600.0);

    // 平行求解所有測試案例
    let mut results: HashMap<String, Grid> = HashMap::new();

    for (task_id, task) in &dataset {
        println!("\n🔍 Processing task: {}", task_id);
        println!("   Training pairs: {}", task.train.len());
        println!("   Test cases: {}", task.test.len());
        debug_log(&format!("Processing task: {} ({} train, {} test)", task_id, task.train.len(), task.test.len()));

        for (test_idx, test_case) in task.test.iter().enumerate() {
            let key = format!("{}_test_{}", task_id, test_idx);
            println!("   Solving test_{}...", test_idx);
            debug_log(&format!("Solving {}", key));
            
            let solution = solver.solve(test_case, &task.train);
            results.insert(key, solution);
        }
    }

    println!("\n✅ All tasks processed!");
    debug_log(&format!("All tasks processed! Total: {}", results.len()));

    // 輸出結果
    println!("\n🏁 Inference complete! Total time: {:?}", start.elapsed());
    println!("   Total predictions: {}", results.len());

    // 轉換為 GridData 並寫入 submission.json
    let submissions: HashMap<String, data::GridData> = results
        .into_iter()
        .map(|(k, v)| (k, data::grid_to_grid_data(&v)))
        .collect();

    if let Err(e) = data::save_submissions(&submissions, output_path) {
        eprintln!("❌ Failed to write submission.json: {}", e);
        debug_log(&format!("Failed to write submission: {}", e));
        std::process::exit(1);
    }
    
    println!("📄 Submission saved to: {}", output_path);
    debug_log(&format!("Submission saved to: {} ({} predictions)", output_path, submissions.len()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = Grid::new(4, 4, 0);
        assert_eq!(grid.height, 4);
        assert_eq!(grid.width, 4);
        assert_eq!(grid.get(0, 0), 0);
    }

    #[test]
    fn test_grid_match() {
        let g1 = Grid::new(3, 3, 1);
        let g2 = Grid::new(3, 3, 1);
        assert_eq!(g1.match_ratio(&g2), 1.0);

        let mut g3 = Grid::new(3, 3, 1);
        g3.set(0, 0, 2);
        assert_eq!(g1.match_ratio(&g3), 8.0 / 9.0);
    }

    #[test]
    fn test_grid_to_objects() {
        let mut grid = Grid::new(4, 4, 0);
        // 畫一個 2x2 的紅色方塊
        for r in 1..3 {
            for c in 1..3 {
                grid.set(r, c, 1);
            }
        }
        let objects = grid_to_objects(&grid);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].color, 1);
        assert_eq!(objects[0].pixels.len(), 4);
    }

    #[test]
    fn test_objects_to_grid() {
        let obj = Object::new(1, vec![(1, 1), (1, 2), (2, 1), (2, 2)]);
        let grid = objects_to_grid(&[obj], 4, 4);
        assert_eq!(grid.get(1, 1), 1);
        assert_eq!(grid.get(0, 0), 0);
    }

    #[test]
    fn test_roundtrip() {
        let mut grid = Grid::new(4, 4, 0);
        grid.set(1, 1, 1);
        grid.set(1, 2, 1);
        grid.set(2, 1, 1);
        grid.set(2, 2, 1);
        grid.set(3, 3, 2);

        let objects = grid_to_objects(&grid);
        let rebuilt = objects_to_grid(&objects, 4, 4);

        assert_eq!(grid.match_ratio(&rebuilt), 1.0);
    }
}
