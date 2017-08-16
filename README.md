Classify target disease of clinical trial project registerred in UMIN-CTR Clinical Trial Registry
================

What is this?
-------------

This is an experimental machine learning project, to train some neural networks to classify texts in the clinical trial registries automatically.

Usage
-----

1.  Download UMIN-CTR comma-separated text archive.

Currently UMIN-ID is required to download the archive. Download files including Japanese, from <http://www.umin.ac.jp/ctr/csvdata.html>

1.  Prepare running environment.

You will need R, {readr} package, and {keras} package.

1.  Generate dataset for training and evaluation.

Source Dataset.R in this repository in your R.

``` r
source("Dataset.R")
```

    ## Warning: Missing column names filled in: 'X110' [110]

    ## Warning: Duplicated column names deduplicated: '試験名/Official
    ## scientific title of the study' => '試験名/Official scientific title of the
    ## study_1' [4], '試験簡略名/Title of the study (Brief title)' => '試験簡略
    ## 名/Title of the study (Brief title)_1' [6], '対象疾患名/Condition' => '対象
    ## 疾患名/Condition_1' [9], '目的1/Narrative objectives1' => '目的1/Narrative
    ## objectives1_1' [14], '目的2 -その他詳細/Basic objectives -Others' => '目
    ## 的2 -その他詳細/Basic objectives -Others_1' [17], '主要アウトカム評価項目/
    ## Primary outcomes' => '主要アウトカム評価項目/Primary outcomes_1' [22], '副
    ## 次アウトカム評価項目/Key secondary outcomes' => '副次アウトカム評価項目/
    ## Key secondary outcomes_1' [24], '介入1/Interventions/Control_1' => '介入1/
    ## Interventions/Control_1_1' [40], '介入2/Interventions/Control_2' => '介入
    ## 2/Interventions/Control_2_1' [42], '介入3/Interventions/Control_3' => '介
    ## 入3/Interventions/Control_3_1' [44], '介入4/Interventions/Control_4' => '介
    ## 入4/Interventions/Control_4_1' [46], '介入5/Interventions/Control_5' => '介
    ## 入5/Interventions/Control_5_1' [48], '介入6/Interventions/Control_6' => '介
    ## 入6/Interventions/Control_6_1' [50], '介入7/Interventions/Control_7' => '介
    ## 入7/Interventions/Control_7_1' [52], '介入8/Interventions/Control_8' => '介
    ## 入8/Interventions/Control_8_1' [54], '介入9/Interventions/Control_9' =>
    ## '介入9/Interventions/Control_9_1' [56], '介入10/Interventions/Control_10'
    ## => '介入10/Interventions/Control_10_1' [58], '選択基準/Key inclusion
    ## criteria' => '選択基準/Key inclusion criteria_1' [63], '除外基準/Key
    ## exclusion criteria' => '除外基準/Key exclusion criteria_1' [65], '研究費
    ## 拠出国/Nationality of Funding Organization' => '研究費拠出国/Nationality
    ## of Funding Organization_1' [77], '共同実施組織/Co-sponsor' => '共同実施
    ## 組織/Co-sponsor_1' [79], 'その他の研究費提供組織/Name of secondary funｄ
    ## er(s)' => 'その他の研究費提供組織/Name of secondary funｄer(s)_1' [81], 'ID
    ## 発行機関1/Org. issuing International ID_1' => 'ID発行機関1/Org. issuing
    ## International ID_1_1' [85], 'ID発行機関2/Org. issuing International ID_2'
    ## => 'ID発行機関2/Org. issuing International ID_2_1' [88], '主な結果/Results'
    ## => '主な結果/Results_1' [103], 'その他関連情報/Other related information'
    ## => 'その他関連情報/Other related information_1' [105]

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_character(),
    ##   `群数/No. of arms` = col_integer(),
    ##   `目標参加者数/Target sample size` = col_integer(),
    ##   `一般公開日（本登録希望日）/Date of disclosure of the study information` = col_date(format = ""),
    ##   `プロトコル確定日/Date of protocol fixation` = col_date(format = ""),
    ##   `登録・組入れ開始（予定）日/Anticipated trial start date` = col_date(format = ""),
    ##   `フォロー終了(予定)日/Last follow-up date` = col_date(format = ""),
    ##   `入力終了(予定)日/Date of closure to data entry` = col_date(format = ""),
    ##   `データ固定（予定）日/Date trial data considered complete` = col_date(format = ""),
    ##   `解析終了(予定)日/Date analysis concluded` = col_date(format = ""),
    ##   `登録日時/Registered date` = col_date(format = ""),
    ##   `最終更新日/Last modified on` = col_date(format = "")
    ## )

    ## See spec(...) for full column specifications.

    ## Warning in rbind(names(probs), probs_f): number of columns of result is not
    ## a multiple of vector length (arg 1)

    ## Warning: 2 parsing failures.
    ## row # A tibble: 2 x 5 col     row                             col   expected      actual expected   <int>                           <chr>      <chr>       <chr> actual 1 15915 目標参加者数/Target sample size an integer  9999999999 file 2 22754 目標参加者数/Target sample size an integer 99999999999 row # ... with 1 more variables: file <chr>

1.  Train the model.

Source train.R in this repository in your R. Currnetly loss is around 0.151, and accuracy is around 0.975.

``` r
source("train.R")
```

1.  Try the trained model.

In training process, you will have `ohmasters` variable that contains dictionary to encode your text. With this, you can test classification by giving some new text.

``` r
eval_title(model_nn, "SGLT2阻害薬", ohmasters$categoryMaster, ohmasters$titlesCharactersMaster, ohmasters$max_title_length)
```

    ## [1] "内科学一般/Medicine in general"                   
    ## [2] "内分泌・代謝病内科学/Endocrinology and Metabolism"
