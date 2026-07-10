use crate::genetic::{ColorRef, GeneAction};
use crate::Grid;

// ============================================================================
// 差異資料結構
// ============================================================================

#[derive(Debug, Clone)]
pub struct PixelDiff {
    pub r: usize,
    pub c: usize,
    pub expected: i32,
    pub actual: i32,
}

#[derive(Debug, Clone)]
pub struct SampleReport {
    pub train_idx: usize,
    pub match_ratio: f64,
    pub errors: Vec<PixelDiff>,
}

#[derive(Debug, Clone)]
pub struct ReflectionReport {
    pub samples: Vec<SampleReport>,
    pub total_errors: usize,
    pub avg_match_ratio: f64,
}

// ============================================================================
// 反思引擎
// ============================================================================

pub struct ReflectionEngine;

impl ReflectionEngine {
    /// 分析單一訓練樣本的預測結果
    pub fn analyze_sample(predicted: &Grid, target: &Grid, train_idx: usize) -> SampleReport {
        let mut errors = Vec::new();
        let mut match_count = 0;
        let total = target.height * target.width;

        for r in 0..target.height {
            for c in 0..target.width {
                let p = predicted.get(r, c);
                let t = target.get(r, c);
                if p == t {
                    match_count += 1;
                } else {
                    errors.push(PixelDiff {
                        r,
                        c,
                        expected: t,
                        actual: p,
                    });
                }
            }
        }

        let match_ratio = match_count as f64 / total as f64;
        SampleReport {
            train_idx,
            match_ratio,
            errors,
        }
    }

    /// 跨樣本分析，生成反思報告
    pub fn analyze_all(
        predictions: &[(Grid, Grid)], // (predicted, target) pairs
    ) -> ReflectionReport {
        let mut samples = Vec::new();
        let mut total_errors = 0;

        for (idx, (predicted, target)) in predictions.iter().enumerate() {
            let report = Self::analyze_sample(predicted, target, idx);
            total_errors += report.errors.len();
            samples.push(report);
        }

        let avg_match_ratio = if samples.is_empty() {
            0.0
        } else {
            samples.iter().map(|s| s.match_ratio).sum::<f64>() / samples.len() as f64
        };

        ReflectionReport {
            samples,
            total_errors,
            avg_match_ratio,
        }
    }

    /// 根據反思報告生成補丁基因
    pub fn generate_patches(report: &ReflectionReport) -> Vec<GeneAction> {
        let mut patches = Vec::new();

        // 策略 0：單一 training sample → 直接修補所有錯誤
        // 多個 training sample 時，只接受「所有樣本都同意」的修正
        if report.total_errors > 0 && report.total_errors <= 10 {
            if report.samples.len() == 1 {
                // 單一樣本：暴力修補
                for sample in &report.samples {
                    for err in &sample.errors {
                        patches.push(GeneAction::FixColorAt {
                            r: err.r,
                            c: err.c,
                            color: ColorRef::Exact(err.expected),
                        });
                    }
                }
                return patches;
            } else {
                // 多個樣本：只接受跨樣本一致的修正
                // 建立 (r, c) → 預期顏色的映射
                let mut pixel_targets: std::collections::HashMap<(usize, usize), std::collections::HashMap<i32, usize>> =
                    std::collections::HashMap::new();
                
                for sample in &report.samples {
                    for err in &sample.errors {
                        pixel_targets
                            .entry((err.r, err.c))
                            .or_insert_with(std::collections::HashMap::new)
                            .entry(err.expected)
                            .and_modify(|c| *c += 1)
                            .or_insert(1);
                    }
                }

                // 只添加所有有錯誤的樣本都同意的修正
                for ((r, c), targets) in &pixel_targets {
                    if targets.len() == 1 {
                        // 所有樣本都同意同一個顏色
                        let (color, count) = targets.iter().max_by_key(|&(_, c)| *c).unwrap();
                        if *count == report.samples.len() {
                            patches.push(GeneAction::FixColorAt {
                                r: *r,
                                c: *c,
                                color: ColorRef::Exact(*color),
                            });
                        }
                    }
                }

                if !patches.is_empty() {
                    return patches;
                }
            }
        }

        // 策略 1：顏色替換分析
        // 收集所有錯誤的 (actual → expected) 顏色映射
        let mut color_map: std::collections::HashMap<i32, std::collections::HashMap<i32, usize>> =
            std::collections::HashMap::new();
        for sample in &report.samples {
            for err in &sample.errors {
                color_map
                    .entry(err.actual)
                    .or_insert_with(std::collections::HashMap::new)
                    .entry(err.expected)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }

        // 找出高頻的顏色映射
        for (from_color, targets) in &color_map {
            let (to_color, count) = targets
                .iter()
                .max_by_key(|&(_, c)| *c)
                .unwrap();
            // 如果這個顏色映射出現在多個錯誤中，生成 ColorSwap
            if *count >= 3 {
                patches.push(GeneAction::ColorSwap {
                    from_color: ColorRef::Exact(*from_color),
                    to_color: ColorRef::Exact(*to_color),
                });
            }
        }

        // 策略 2：區塊填充分析
        // 如果某個樣本有大量連續錯誤，嘗試 Fill
        for sample in &report.samples {
            if sample.errors.len() > 10 {
                // 找出錯誤的邊界框
                let min_r = sample.errors.iter().map(|e| e.r).min().unwrap_or(0);
                let max_r = sample.errors.iter().map(|e| e.r).max().unwrap_or(0);
                let min_c = sample.errors.iter().map(|e| e.c).min().unwrap_or(0);
                let max_c = sample.errors.iter().map(|e| e.c).max().unwrap_or(0);

                // 統計邊界框中最常見的預期顏色
                let mut color_counts: std::collections::HashMap<i32, usize> =
                    std::collections::HashMap::new();
                for err in &sample.errors {
                    *color_counts.entry(err.expected).or_insert(0) += 1;
                }

                if let Some((fill_color, _)) = color_counts
                    .iter()
                    .max_by_key(|&(_, c)| *c)
                {
                    let h = max_r.saturating_sub(min_r) + 1;
                    let w = max_c.saturating_sub(min_c) + 1;
                    if h > 1 || w > 1 {
                        patches.push(GeneAction::Fill {
                            color: ColorRef::Exact(*fill_color),
                            r: min_r,
                            c: min_c,
                            h,
                            w,
                        });
                    }
                }
            }
        }

        patches
    }

    /// 將補丁應用到基因體
    pub fn apply_patches(genome: &mut crate::genetic::Genome, patches: &[GeneAction]) {
        for patch in patches {
            genome.actions.push(patch.clone());
        }
    }

    /// 判斷是否需要觸發反思
    pub fn should_reflect(report: &ReflectionReport, target_fitness: f64) -> bool {
        // 如果還沒達到目標，且總錯誤數不多 (有希望)
        report.avg_match_ratio < target_fitness && report.total_errors <= 20
    }
}

// ============================================================================
// 單元測試
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;

    fn make_grid(height: usize, width: usize, fill: i32) -> Grid {
        Grid::new(height, width, fill)
    }

    #[test]
    fn test_analyze_sample_perfect() {
        let predicted = make_grid(3, 3, 1);
        let target = make_grid(3, 3, 1);
        let report = ReflectionEngine::analyze_sample(&predicted, &target, 0);
        assert_eq!(report.match_ratio, 1.0);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_analyze_sample_with_errors() {
        let mut predicted = make_grid(3, 3, 0);
        predicted.set(0, 0, 1);
        predicted.set(1, 1, 2);

        let target = make_grid(3, 3, 0);

        let report = ReflectionEngine::analyze_sample(&predicted, &target, 0);
        assert_eq!(report.errors.len(), 2);
        assert_eq!(report.match_ratio, 7.0 / 9.0);
    }

    #[test]
    fn test_generate_patches_micro() {
        let mut predicted = make_grid(3, 3, 0);
        let mut target = make_grid(3, 3, 0);
        target.set(0, 0, 5);
        target.set(1, 1, 3);

        let predictions = vec![(predicted, target)];
        let report = ReflectionEngine::analyze_all(&predictions);
        let patches = ReflectionEngine::generate_patches(&report);

        // 應該生成 2 個 FixColorAt 補丁
        assert_eq!(patches.len(), 2);
        assert!(matches!(patches[0], GeneAction::FixColorAt { .. }));
    }

    #[test]
    fn test_generate_patches_color_swap() {
        let mut predicted = make_grid(4, 4, 1);
        let mut target = make_grid(4, 4, 2);

        let predictions = vec![(predicted, target)];
        let report = ReflectionEngine::analyze_all(&predictions);
        let patches = ReflectionEngine::generate_patches(&report);

        // 16 個錯誤都是 1→2，應該生成 ColorSwap (> 10 錯誤所以不走 micro-patch)
        assert!(patches.iter().any(|p| matches!(p, GeneAction::ColorSwap { .. })));
    }

    #[test]
    fn test_should_reflect() {
        let report = ReflectionReport {
            samples: vec![SampleReport {
                train_idx: 0,
                match_ratio: 0.9,
                errors: vec![PixelDiff {
                    r: 0,
                    c: 0,
                    expected: 1,
                    actual: 0,
                }],
            }],
            total_errors: 1,
            avg_match_ratio: 0.9,
        };
        assert!(ReflectionEngine::should_reflect(&report, 0.999));
    }

    #[test]
    fn test_should_not_reflect_when_perfect() {
        let report = ReflectionReport {
            samples: vec![SampleReport {
                train_idx: 0,
                match_ratio: 1.0,
                errors: vec![],
            }],
            total_errors: 0,
            avg_match_ratio: 1.0,
        };
        assert!(!ReflectionEngine::should_reflect(&report, 0.999));
    }
}
