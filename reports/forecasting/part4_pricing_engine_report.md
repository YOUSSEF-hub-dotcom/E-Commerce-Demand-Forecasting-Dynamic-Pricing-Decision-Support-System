# Part-4 Pricing Engine Report (Dynamic Pricing)

* Generated: 2026-09-24T15:19:44.985680+00:00
* Reference week: `2018-05-14` (Part-4 VAL period, test untouched: False)
* Cost source: `NONE - profit NOT computable`

## Summary

| Metric | value |
|---|---|
| products_scored | 1385 |
| products_skipped_no_price | 31344 |
| products_with_baseline_price | 1385 |
| products_with_effective_elasticity | 468 |
| products_recommended_optimized | 0 |
| products_price_recommended_change | 0 |
| products_profit_computable | 0 |
| products_extrapolation_flagged | 468 |

| Guardrail | applied count |
|---|---|
| margin_floor_applied | 0 |
| margin_floor_inactive_no_cost | 0 |
| ceiling_applied | 16 |
| floor_applied | 0 |
| smoothed | 0 |

## Elasticity (pooled, product FE)
- pooled log-log b=-0.3952, se=0.0363, t=-10.878, n=41452, R2=0.0053
- **Descriptive** demand response; no causal claim (no experimental/instrumental price variation).

## Recommended-price sample (top 10 absolute change)
```
                      product_id              category                                   reason elasticity_source  current_price  optimal_price  recommended_price objective
87feb07adc221a4c6cdf051ea1afd0ff    eletrodomesticos_2 PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled        2110.00       2532.000            2532.00   revenue
16c4e87b98a9370a9cbc3a4658a3f45b instrumentos_musicais PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled        1899.00       2278.800            2278.80   revenue
89f59eee27a6a49b9f0d887136c71cd4          beleza_saude PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled        1299.65       1559.580            1559.58   revenue
5237739bb5fee495dbd337755a138660    relogios_presentes PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled         879.00       1054.800            1054.80   revenue
16d928f7b23b3d30bd5a9ff281eed9b8             telefonia PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled         849.00       1018.800            1018.80   revenue
85e8e8aad6ad12d67c21300379760196    ferramentas_jardim PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled        1186.00        948.800             948.80   revenue
af8533a77adce26dca87480c4c5c9af4            cool_stuff PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled         699.00        838.800             838.80   revenue
3b04dbc12a3fcaa8402b8dd2b3a909d7    ferramentas_jardim PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled         989.00        791.200             791.20   revenue
34b91723112fa9ebf6b4d6b4db4a6741            brinquedos PROFIT_NOT_COMPUTABLE (no cost supplied)               own         599.99        719.988             719.99   revenue
69f07c49379a0c4bd660dd794a72f02a            cool_stuff PROFIT_NOT_COMPUTABLE (no cost supplied)   category_pooled         599.90        719.880             719.88   revenue
```

> Profit optimization is NOT executable until a valid cost source is supplied; until then
> the revenue-optimal scenario is returned and flagged `PROFIT_NOT_COMPUTABLE`. Elasticity
> estimates are descriptive, not causal.