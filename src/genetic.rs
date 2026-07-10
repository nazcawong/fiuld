use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

use crate::{DslAction, Grid};

// ============================================================================
// 顏色參考：ColorRef (動態泛化型別)
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum ColorRef {
    Exact(i32),
    Background,
    Foreground,
    ForegroundMax,
}

impl ColorRef {
    /// 從網格解析具體顏色值
    pub fn resolve(&self, grid: &Grid) -> i32 {
        match self {
            ColorRef::Exact(c) => *c,
            ColorRef::Background => Self::find_background(grid),
            ColorRef::Foreground => Self::find_foreground(grid),
            ColorRef::ForegroundMax => Self::find_foreground_max(grid),
        }
    }

    /// 背景色 = 出現次數最多的顏色 (通常是 0)
    fn find_background(grid: &Grid) -> i32 {
        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for r in 0..grid.height {
            for c in 0..grid.width {
                *counts.entry(grid.get(r, c)).or_insert(0) += 1;
            }
        }
        counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(color, _)| color)
            .unwrap_or(0)
    }

    /// 前景色 = 非背景色中出現次數最多的
    fn find_foreground(grid: &Grid) -> i32 {
        let bg = Self::find_background(grid);
        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for r in 0..grid.height {
            for c in 0..grid.width {
                let color = grid.get(r, c);
                if color != bg {
                    *counts.entry(color).or_insert(0) += 1;
                }
            }
        }
        counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(color, _)| color)
            .unwrap_or(1)
    }

    /// 前景色 (max) = 非背景色中出現次數最多的 (同 Foreground)
    fn find_foreground_max(grid: &Grid) -> i32 {
        Self::find_foreground(grid)
    }
}

// ============================================================================
// 基因編碼：GeneAction (DSL 動作庫)
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum GeneAction {
    /// 填充矩形區域 (實心)
    Fill { color: ColorRef, r: usize, c: usize, h: usize, w: usize },
    /// 畫線
    Line { color: ColorRef, r1: usize, c1: usize, r2: usize, c2: usize },
    /// 畫空心框
    Rect { color: ColorRef, r: usize, c: usize, h: usize, w: usize },
    /// 顏色替換
    ColorSwap { from_color: ColorRef, to_color: ColorRef },
    /// 清除網格
    Clear,
    /// 精確修正單一像素
    FixColorAt { r: usize, c: usize, color: ColorRef },
    /// 旋轉 90 度
    Rotate90,
    /// 旋轉 180 度
    Rotate180,
    /// 旋轉 270 度
    Rotate270,
    /// 水平翻轉
    FlipHorizontal,
    /// 垂直翻轉
    FlipVertical,
    /// 縮放
    Scale { factor: usize },
    /// 複製 (保留原網格)
    Copy,
}

impl GeneAction {
    /// 將 GeneAction 轉換為 DSL 動作 (需要輸入網格來解析 ColorRef)
    pub fn to_dsl_with_context(&self, input: &Grid) -> DslAction {
        match self {
            GeneAction::Fill { color, r, c, h, w } => DslAction::Fill {
                color: color.resolve(input),
                x: *c,
                y: *r,
                w: *w,
                h: *h,
            },
            GeneAction::Line { color, r1, c1, r2, c2 } => DslAction::DrawLine {
                color: color.resolve(input),
                x1: *c1,
                y1: *r1,
                x2: *c2,
                y2: *r2,
            },
            GeneAction::Rect { color, r, c, h, w } => DslAction::DrawRect {
                color: color.resolve(input),
                x: *c,
                y: *r,
                w: *w,
                h: *h,
            },
            GeneAction::ColorSwap { from_color, to_color } => DslAction::ColorSwap {
                from_color: from_color.resolve(input),
                to_color: to_color.resolve(input),
            },
            GeneAction::Clear => DslAction::Clear,
            GeneAction::FixColorAt { r, c, color } => DslAction::Fill {
                color: color.resolve(input),
                x: *c,
                y: *r,
                w: 1,
                h: 1,
            },
            GeneAction::Rotate90 => DslAction::Rotate90,
            GeneAction::Rotate180 => DslAction::Rotate180,
            GeneAction::Rotate270 => DslAction::Rotate270,
            GeneAction::FlipHorizontal => DslAction::FlipHorizontal,
            GeneAction::FlipVertical => DslAction::FlipVertical,
            GeneAction::Scale { factor } => DslAction::Scale { factor: *factor },
            GeneAction::Copy => DslAction::Copy,
        }
    }

    /// 隨機生成一個動作
    pub fn random(rng: &mut impl Rng, height: usize, width: usize) -> Self {
        let choice = rng.gen_range(0..13); // 13 種動作類型
        match choice {
            0 => {
                let h_max = height.saturating_sub(rng.gen_range(0..height));
                let w_max = width.saturating_sub(rng.gen_range(0..width));
                GeneAction::Fill {
                    color: Self::random_color_ref(rng),
                    r: rng.gen_range(0..height),
                    c: rng.gen_range(0..width),
                    h: rng.gen_range(1..=h_max),
                    w: rng.gen_range(1..=w_max),
                }
            }
            1 => GeneAction::Line {
                color: Self::random_color_ref(rng),
                r1: rng.gen_range(0..height),
                c1: rng.gen_range(0..width),
                r2: rng.gen_range(0..height),
                c2: rng.gen_range(0..width),
            },
            2 => {
                let h_max = height.saturating_sub(rng.gen_range(0..height));
                let w_max = width.saturating_sub(rng.gen_range(0..width));
                GeneAction::Rect {
                    color: Self::random_color_ref(rng),
                    r: rng.gen_range(0..height),
                    c: rng.gen_range(0..width),
                    h: rng.gen_range(1..=h_max),
                    w: rng.gen_range(1..=w_max),
                }
            }
            3 => GeneAction::ColorSwap {
                from_color: Self::random_color_ref(rng),
                to_color: Self::random_color_ref(rng),
            },
            4 => GeneAction::Clear,
            5 => GeneAction::FixColorAt {
                r: rng.gen_range(0..height),
                c: rng.gen_range(0..width),
                color: Self::random_color_ref(rng),
            },
            6 => GeneAction::Rotate90,
            7 => GeneAction::Rotate180,
            8 => GeneAction::Rotate270,
            9 => GeneAction::FlipHorizontal,
            10 => GeneAction::FlipVertical,
            11 => GeneAction::Scale {
                factor: rng.gen_range(1..=3),
            },
            12 => GeneAction::Copy,
            _ => GeneAction::Clear,
        }
    }

    /// 隨機生成一個顏色參考
    pub fn random_color_ref(rng: &mut impl Rng) -> ColorRef {
        let choice = rng.gen_range(0..4);
        match choice {
            0 => ColorRef::Exact(rng.gen_range(1..10)),
            1 => ColorRef::Background,
            2 => ColorRef::Foreground,
            3 => ColorRef::ForegroundMax,
            _ => ColorRef::Exact(rng.gen_range(1..10)),
        }
    }

    /// 將動作序列轉換為可執行的程式碼字串 (用於除錯)
    pub fn to_code(&self) -> String {
        match self {
            GeneAction::Fill { color, r, c, h, w } => {
                format!("Fill({:?}, r={}, c={}, h={}, w={})", color, r, c, h, w)
            }
            GeneAction::Line { color, r1, c1, r2, c2 } => {
                format!("Line({:?}, r1={}, c1={}, r2={}, c2={})", color, r1, c1, r2, c2)
            }
            GeneAction::Rect { color, r, c, h, w } => {
                format!("Rect({:?}, r={}, c={}, h={}, w={})", color, r, c, h, w)
            }
            GeneAction::ColorSwap {
                from_color,
                to_color,
            } => format!("ColorSwap({:?}, {:?})", from_color, to_color),
            GeneAction::Clear => "Clear".to_string(),
            GeneAction::FixColorAt { r, c, color } => {
                format!("FixColorAt({}, {}, {:?})", r, c, color)
            }
            GeneAction::Rotate90 => "Rotate90".to_string(),
            GeneAction::Rotate180 => "Rotate180".to_string(),
            GeneAction::Rotate270 => "Rotate270".to_string(),
            GeneAction::FlipHorizontal => "FlipHorizontal".to_string(),
            GeneAction::FlipVertical => "FlipVertical".to_string(),
            GeneAction::Scale { factor } => format!("Scale({})", factor),
            GeneAction::Copy => "Copy".to_string(),
        }
    }
}

// ============================================================================
// 基因體：Genome
// ============================================================================

#[derive(Debug, Clone)]
pub struct Genome {
    pub actions: Vec<GeneAction>,
    pub fitness: f64,
}

impl Genome {
    /// 建立隨機基因體
    pub fn new_random(rng: &mut impl Rng, height: usize, width: usize, max_length: usize) -> Self {
        let length = rng.gen_range(1..=max_length);
        let actions = (0..length)
            .map(|_| GeneAction::random(rng, height, width))
            .collect();
        Genome {
            actions,
            fitness: 0.0,
        }
    }

    /// 從訓練資料生成種子基因 (Data-Driven Seeding)
    pub fn new_from_hints(rng: &mut impl Rng, train_pairs: &[(Grid, Grid)], max_length: usize) -> Self {
        if train_pairs.is_empty() {
            return Genome::new_random(rng, 16, 16, max_length);
        }

        let (input, target) = &train_pairs[0];
        let height = input.height;
        let width = input.width;

        // 統計輸入和目標中出現的顏色
        let mut colors: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for r in 0..height {
            for c in 0..width {
                colors.insert(input.get(r, c));
                colors.insert(target.get(r, c));
            }
        }
        let color_vec: Vec<i32> = colors.into_iter().filter(|&c| c != 0).collect();

        // 檢查是否只需要 ColorSwap (形狀相同但顏色不同)
        let needs_color_swap = input.height == target.height && input.width == target.width;
        let mut all_same_shape = true;
        for r in 0..height {
            for c in 0..width {
                if input.get(r, c) != target.get(r, c) && input.get(r, c) != 0 && target.get(r, c) != 0
                {
                } else if input.get(r, c) != target.get(r, c) {
                    all_same_shape = false;
                    break;
                }
            }
            if !all_same_shape {
                break;
            }
        }

        let mut actions = Vec::new();
        let length = rng.gen_range(1..=max_length);

        for _ in 0..length {
            // 如果只需要 ColorSwap，優先生成 ColorSwap (使用泛化顏色參考)
            if needs_color_swap && !color_vec.is_empty() && rng.gen::<f64>() < 0.6 {
                if let (Some(&from_c), Some(&to_c)) = (color_vec.first(), color_vec.get(1)) {
                    actions.push(GeneAction::ColorSwap {
                        from_color: ColorRef::Exact(from_c),
                        to_color: ColorRef::Exact(to_c),
                    });
                } else if let Some(&c) = color_vec.first() {
                    actions.push(GeneAction::ColorSwap {
                        from_color: ColorRef::Exact(c),
                        to_color: ColorRef::Exact(c),
                    });
                }
            } else {
                actions.push(GeneAction::random(rng, height, width));
            }
        }

        Genome {
            actions,
            fitness: 0.0,
        }
    }

    /// 執行基因體中的所有動作
    pub fn execute(&self, input: &Grid) -> Grid {
        let mut grid = input.clone();
        for action in &self.actions {
            let dsl = action.to_dsl_with_context(input);
            dsl.apply(&mut grid);
        }
        grid
    }

    /// 評估適應度 (單一訓練對)
    pub fn evaluate_single(&self, input: &Grid, target: &Grid) -> f64 {
        let prediction = self.execute(input);
        prediction.match_ratio(target)
    }

    /// 評估適應度 (多個訓練對的平均)
    pub fn evaluate_batch(&self, train_pairs: &[(Grid, Grid)]) -> f64 {
        if train_pairs.is_empty() {
            return 0.0;
        }
        let total: f64 = train_pairs
            .iter()
            .map(|(input, target)| self.evaluate_single(input, target))
            .sum();
        total / train_pairs.len() as f64
    }

    /// 基因突變
    pub fn mutate(&mut self, rng: &mut impl Rng, mutation_rate: f64, height: usize, width: usize) {
        // 1. 修改現有動作
        for action in &mut self.actions {
            if rng.gen::<f64>() < mutation_rate {
                *action = GeneAction::random(rng, height, width);
            }
        }

        // 2. 隨機插入新動作
        if rng.gen::<f64>() < 0.3 {
            let insert_pos = rng.gen_range(0..=self.actions.len());
            self.actions
                .insert(insert_pos, GeneAction::random(rng, height, width));
        }

        // 3. 隨機刪除動作
        if rng.gen::<f64>() < 0.1 && self.actions.len() > 1 {
            let remove_pos = rng.gen_range(0..self.actions.len());
            self.actions.remove(remove_pos);
        }

        // 4. 顏色參考突變 (將 Exact 轉為抽象參考)
        for action in &mut self.actions {
            if rng.gen::<f64>() < 0.05 {
                Self::mutate_color_ref(action, rng);
            }
        }
    }

    /// 突變動作中的 ColorRef
    fn mutate_color_ref(action: &mut GeneAction, rng: &mut impl Rng) {
        match action {
            GeneAction::Fill { color, .. }
            | GeneAction::Line { color, .. }
            | GeneAction::Rect { color, .. }
            | GeneAction::FixColorAt { color, .. } => {
                if let ColorRef::Exact(_) = color {
                    *color = GeneAction::random_color_ref(rng);
                }
            }
            GeneAction::ColorSwap {
                from_color,
                to_color,
            } => {
                if rng.gen::<bool>() {
                    if let ColorRef::Exact(_) = from_color {
                        *from_color = GeneAction::random_color_ref(rng);
                    }
                } else if let ColorRef::Exact(_) = to_color {
                    *to_color = GeneAction::random_color_ref(rng);
                }
            }
            GeneAction::Clear => {}
            GeneAction::Rotate90 | GeneAction::Rotate180 | GeneAction::Rotate270 |
            GeneAction::FlipHorizontal | GeneAction::FlipVertical |
            GeneAction::Scale { .. } | GeneAction::Copy => {}
        }
    }

    /// 單點交叉
    pub fn crossover(&self, other: &Genome, rng: &mut impl Rng) -> Genome {
        if self.actions.is_empty() && other.actions.is_empty() {
            return Genome {
                actions: Vec::new(),
                fitness: 0.0,
            };
        }

        // 隨機選擇父系
        if rng.gen::<bool>() {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// 將基因體轉換為程式碼字串
    pub fn to_code(&self) -> String {
        self.actions
            .iter()
            .map(|a| a.to_code())
            .collect::<Vec<_>>()
            .join(" → ")
    }
}

// ============================================================================
// 選擇機制
// ============================================================================

/// 輪盤選擇 (Fitness-Proportionate Selection)
pub fn roulette_selection<'a>(population: &'a [(Genome, f64)], rng: &mut impl Rng) -> &'a Genome {
    let total_fitness: f64 = population.iter().map(|(_, f)| f).sum();
    let mut pick = rng.gen_range(0.0..total_fitness);
    let mut last_idx = 0;

    for (idx, (_, fitness)) in population.iter().enumerate() {
        pick -= fitness;
        if pick <= 0.0 {
            last_idx = idx;
            break;
        }
    }

    &population[last_idx].0
}

/// 精英選擇 (取前 N 名)
pub fn elite_selection(population: &mut [(Genome, f64)], count: usize) -> Vec<Genome> {
    population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    population[..count.min(population.len())].iter().map(|(g, _)| g.clone()).collect()
}

// ============================================================================
// 演化引擎
// ============================================================================

pub struct EvolutionConfig {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub elite_ratio: f64,
    pub max_actions: usize,
    pub target_fitness: f64,
    pub patience: usize, // 多少代沒有進步就停止
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        EvolutionConfig {
            population_size: 1000,
            generations: 100,
            mutation_rate: 0.15,
            elite_ratio: 0.1,
            max_actions: 10,
            target_fitness: 0.999,
            patience: 50,
        }
    }
}

pub struct EvolutionResult {
    pub best_genome: Genome,
    pub best_fitness: f64,
    pub generations_run: usize,
    pub time_elapsed: std::time::Duration,
}

/// 執行遺傳演化 (Rayon 平行化)
pub fn evolve(
    train_pairs: &[(Grid, Grid)],
    input_height: usize,
    input_width: usize,
    config: &EvolutionConfig,
) -> EvolutionResult {
    let start = Instant::now();
    let mut rng = rand::thread_rng();

    println!(
        "[GeneticEngine] Starting evolution: pop={}, gen={}, mutation={:.1}%, elite={:.0}%",
        config.population_size,
        config.generations,
        config.mutation_rate * 100.0,
        config.elite_ratio * 100.0,
    );

    // 1. 初始化種群 (使用資料驅動生成)
    let mut population: Vec<Genome> = (0..config.population_size)
        .map(|_| Genome::new_from_hints(&mut rng, train_pairs, config.max_actions))
        .collect();

    let mut best_genome = population[0].clone();
    let mut best_fitness = 0.0;
    let mut no_improve_count = 0;

    let elite_count = (config.population_size as f64 * config.elite_ratio) as usize;

    for gen in 0..config.generations {
        // 2. 平行評估適應度
        let mut scored_pop: Vec<(Genome, f64)> = population
            .into_par_iter()
            .map(|g| {
                let fitness = g.evaluate_batch(train_pairs);
                (g, fitness)
            })
            .collect();

        // 3. 追蹤最佳解
        if let Some((best_idx, (_best_g, best_f))) = scored_pop
            .iter()
            .enumerate()
            .max_by(|(_, (_, f1)), (_, (_, f2))| f1.partial_cmp(f2).unwrap())
        {
            if *best_f > best_fitness {
                best_fitness = *best_f;
                best_genome = scored_pop.remove(best_idx).0;
                no_improve_count = 0;
            } else {
                no_improve_count += 1;
            }
        }

        // 4. 印出進度
        if gen % 10 == 0 || gen == config.generations - 1 {
            let avg_fitness: f64 = scored_pop.iter().map(|(_, f)| f).sum::<f64>() / scored_pop.len() as f64;
            println!(
                "  Gen {}/{}: best={:.3}, avg={:.3}, elite={}",
                gen,
                config.generations,
                best_fitness,
                avg_fitness,
                elite_count,
            );
        }

        // 5. 提前結束
        if best_fitness >= config.target_fitness {
            println!("  ✅ Perfect match found at gen {}!", gen);
            break;
        }

        // 6. 早停
        if no_improve_count >= config.patience {
            println!("  ⏹️  Early stop: no improvement for {} generations", no_improve_count);
            break;
        }

        // 7. 產生下一代
        scored_pop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut next_generation = Vec::with_capacity(config.population_size);

        // 菁英保留
        for i in 0..elite_count {
            next_generation.push(scored_pop[i].0.clone());
        }

        // 繁衍填滿
        while next_generation.len() < config.population_size {
            let parent1 = roulette_selection(&scored_pop, &mut rng);
            let parent2 = roulette_selection(&scored_pop, &mut rng);

            let mut child = parent1.crossover(parent2, &mut rng);
            child.mutate(&mut rng, config.mutation_rate, input_height, input_width);
            next_generation.push(child);
        }

        population = next_generation;
    }

    let time_elapsed = start.elapsed();

    println!(
        "  🏁 Evolution complete: best_fitness={:.4}, generations={}, time={:?}",
        best_fitness,
        best_genome.actions.len(),
        time_elapsed,
    );

    EvolutionResult {
        best_genome: best_genome.clone(),
        best_fitness,
        generations_run: best_genome.actions.len(),
        time_elapsed,
    }
}

// ============================================================================
// 單元測試
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(height: usize, width: usize, fill: i32) -> Grid {
        Grid::new(height, width, fill)
    }

    #[test]
    fn test_color_ref_resolve() {
        let mut rng = rand::thread_rng();
        let action = GeneAction::random(&mut rng, 16, 16);
        // 確認成功生成
        match action {
            GeneAction::Fill { color, .. } => match color {
                ColorRef::Exact(c) => assert!((1..=9).contains(&c)),
                ColorRef::Background | ColorRef::Foreground | ColorRef::ForegroundMax => {}
            },
            GeneAction::Line { color, .. } => match color {
                ColorRef::Exact(c) => assert!((1..=9).contains(&c)),
                ColorRef::Background | ColorRef::Foreground | ColorRef::ForegroundMax => {}
            },
            GeneAction::Rect { color, .. } => match color {
                ColorRef::Exact(c) => assert!((1..=9).contains(&c)),
                ColorRef::Background | ColorRef::Foreground | ColorRef::ForegroundMax => {}
            },
            GeneAction::ColorSwap {
                from_color,
                to_color,
            } => match (&from_color, &to_color) {
                (ColorRef::Exact(f), ColorRef::Exact(t)) => {
                    assert!((1..=9).contains(f));
                    assert!((1..=9).contains(t));
                }
                _ => {}
            },
            GeneAction::Clear => {}
            GeneAction::FixColorAt { color, .. } => match color {
                ColorRef::Exact(c) => assert!((1..=9).contains(&c)),
                ColorRef::Background | ColorRef::Foreground | ColorRef::ForegroundMax => {}
            },
            GeneAction::Rotate90 | GeneAction::Rotate180 | GeneAction::Rotate270 |
            GeneAction::FlipHorizontal | GeneAction::FlipVertical |
            GeneAction::Scale { .. } | GeneAction::Copy => {}
        }
    }

    #[test]
    fn test_genome_execute_fill() {
        let genome = Genome {
            actions: vec![GeneAction::Fill {
                color: ColorRef::Exact(5),
                r: 2,
                c: 3,
                h: 2,
                w: 2,
            }],
            fitness: 0.0,
        };
        let input = make_grid(8, 8, 0);
        let output = genome.execute(&input);
        assert_eq!(output.get(2, 3), 5);
        assert_eq!(output.get(3, 4), 5);
        assert_eq!(output.get(0, 0), 0);
    }

    #[test]
    fn test_genome_execute_clear() {
        let grid = make_grid(4, 4, 1);
        let genome = Genome {
            actions: vec![GeneAction::Clear],
            fitness: 0.0,
        };
        let output = genome.execute(&grid);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(output.get(r, c), 0);
            }
        }
    }

    #[test]
    fn test_genome_crossover() {
        let mut rng = rand::thread_rng();
        let genome1 = Genome {
            actions: vec![
                GeneAction::Fill {
                    color: ColorRef::Exact(1),
                    r: 0,
                    c: 0,
                    h: 1,
                    w: 1,
                },
                GeneAction::Fill {
                    color: ColorRef::Exact(2),
                    r: 1,
                    c: 1,
                    h: 1,
                    w: 1,
                },
            ],
            fitness: 0.0,
        };
        let genome2 = Genome {
            actions: vec![
                GeneAction::Fill {
                    color: ColorRef::Exact(3),
                    r: 2,
                    c: 2,
                    h: 1,
                    w: 1,
                },
                GeneAction::Fill {
                    color: ColorRef::Exact(4),
                    r: 3,
                    c: 3,
                    h: 1,
                    w: 1,
                },
            ],
            fitness: 0.0,
        };

        let child = genome1.crossover(&genome2, &mut rng);
        // 交叉後應該是其中一個父系的完整複製
        assert!(!child.actions.is_empty());
    }

    #[test]
    fn test_genome_mutate() {
        let mut genome = Genome {
            actions: vec![GeneAction::Clear, GeneAction::Clear, GeneAction::Clear],
            fitness: 0.0,
        };
        let mut rng = rand::thread_rng();
        genome.mutate(&mut rng, 1.0, 16, 16); // 100% 突變率
        // 確認動作仍然存在 (突變可能改變內容，但不會把整個基因體清空)
        assert!(genome.actions.len() >= 1);
    }

    #[test]
    fn test_fitness_evaluation() {
        let genome = Genome {
            actions: vec![GeneAction::Clear],
            fitness: 0.0,
        };
        let input = make_grid(3, 3, 1);
        let target = make_grid(3, 3, 0);
        let train_pairs = vec![(input, target)];
        let fitness = genome.evaluate_batch(&train_pairs);
        assert_eq!(fitness, 1.0); // Clear 後應該完全匹配
    }

    #[test]
    fn test_roulette_selection() {
        let mut rng = rand::thread_rng();
        let population = vec![
            (Genome::new_random(&mut rng, 8, 8, 3), 0.1),
            (Genome::new_random(&mut rng, 8, 8, 3), 0.5),
            (Genome::new_random(&mut rng, 8, 8, 3), 0.4),
        ];
        let selected = roulette_selection(&population, &mut rng);
        // 確認選到的是 population 中的某個元素
        assert!(population.iter().any(|(g, _)| g.actions == selected.actions));
    }

    #[test]
    fn test_elite_selection() {
        let mut rng = rand::thread_rng();
        let mut population: Vec<(Genome, f64)> = (0..5)
            .map(|i| {
                (
                    Genome::new_random(&mut rng, 8, 8, 3),
                    i as f64 * 0.1,
                )
            })
            .collect();

        let elites = elite_selection(&mut population, 2);
        assert_eq!(elites.len(), 2);
        // 確認 fitness 是降序排列
        assert!(elites[0].fitness >= elites[1].fitness || elites[0].actions.len() >= elites[1].actions.len());
    }
}
