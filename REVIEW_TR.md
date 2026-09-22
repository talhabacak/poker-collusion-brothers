# Tarık için kontrol listesi

Takım içi not (Türkçe). Depo public olmadan önce silinebilir; kalırsa da bir zararı
yok, yalnızca neyin kim tarafından doğrulandığını gösterir.

Bu depo, ödül uygunluğu için organizatörün 21 Eylül duyurusunda istediği üç şeyi
karşılamak üzere kuruldu: ≤1500 kelimelik write-up, seçilen gönderimi yeniden üreten
kod, ve beş kanıt vaka incelemesi. **Yazılacak bir şey kalmadı; aşağıdakiler yalnızca
doğrulama.** Katılmadığın bir cümle olursa söyle, düzeltirim.

---

## 1. Kodun senin kodun mu (2 dakika)

`pipeline/scripts/` içindeki base zincirinin 13 dosyası, v55'i kuran commit'ten
(`823dce0`) birebir alındı. Yarışma reposunda, `tarik` dalı çekiliyken:

```bash
git fetch origin tarik
for f in 01_prep 02_policy 03_pairhand 03b_relational 03c_extra 13_oppaware \
         15_oppcond_policy 04_hand_scorer 04c_action_model 04b_stage1b \
         05b_evidence_reranker 06_validate_submission; do
  diff <(tr -d '\r' < <REPO>/pipeline/scripts/$f.py) <(git show 823dce0:src/$f.py | tr -d '\r') \
    > /dev/null && echo "AYNI $f" || echo "FARKLI $f"
done
diff <(tr -d '\r' < <REPO>/pipeline/scripts/05_pair_risk_behaviour.py) \
     <(git show 823dce0:src/05_pair_model.py | tr -d '\r')     # 0 fark beklenir
```

- Tek istisna **`01_prep.py`**: Windows'ta `Pool` worker'ları fork yerine spawn ettiği
  için betiğin gövdesi `main()` + `if __name__ == "__main__"` altına alındı. Docstring'de
  yazıyor. Özellik, seed, filtre, sabit **değişmedi** — bunu bir gözden geçirmeni
  isterim, tek "kod" değişikliği bu.
- `05_pair_model.py` → **`05_pair_risk_behaviour.py`** olarak yeniden adlandırıldı,
  çünkü risk zincirinde de aynı adlı bir dosya var. İçerik aynı.
- `05b_mil.py` senin reranker'ının kancalı kopyası (stage 533 ölçüm için kullanıyor);
  kancasız koşunca 0.66085'i birebir veriyor.

## 2. Bayraklar doğru mu (1 dakika)

`run_all.py`'nin `base` fazı `docs/base_chain/run_A3.sh`'i satır satır takip ediyor:

```
U_WEIGHT=0.0 MIXED_NEG_W=0 USE_VAL=0 USE_CT=0 USE_CTR=0 USE_OA=1 USE_OC=1 OA_COLS=""
USE_OAH=1 USE_OCH=1 RERANK_SEEDS=42,7,2024,11,99,5
SUB_OUT=submission_v55_base.csv → RERANK_IN=submission_v55_base.csv
```

```bash
python run_all.py --dry-run --with-base | grep base1[12]
```

`10_ambiguous_family.py` (dördüncü aile kuralı) **yok**: v55 onsuz, run_A3 onu ayrı bir
dosya olarak üretiyordu.

## 3. Senin kayıtların (5 dakika)

`docs/base_chain/` altında, `edddbb5`'ten birebir: `AUTHOR_README.md`,
`EXPERIMENTS.md`, `SOLUTION_WRITEUP.md`, `case_reviews.md` ve v55 soyağacındaki altı
run betiği. Klasörün kendi README'si, `SOLUTION_WRITEUP.md`'nin v33/v34 zincirini
anlattığı için **güncel olmadığını** ve vaka incelemelerinin seçilmeyen bir dosyadan
geldiğini açıkça yazıyor. Bu ikisini "eski" diye işaretlememe itirazın var mı?

## 4. Senin zincirin hakkında yazdığım iddialar (asıl kontrol edilecek yer)

Yanlış ya da fazla iddialı bulduğun varsa söyle:

| nerede | iddia |
|---|---|
| `WRITEUP.md` | rakip-farkında bloğun taşınması tahtayı 0.90858 → 0.91748 (+0.0089) taşıdı |
| `WRITEUP.md` | risk/davranış LightGBM, etiketsiz pair ağırlığı 0, kanıt lambdarank 6 tohum, top-50 aday |
| `docs/TWO_CHAINS.md` | birleşmede senin en iyin v45 0.90858, benimki v41/v44 0.90731 — 0.0013 fark |
| `docs/TWO_CHAINS.md` | birleşme sonrası her dosyanın kompozisyonu ve public/private skorları |
| `docs/MEASUREMENT_NOTES.md` | geç gönderilen v58 0.92121/0.92288, v58_amb 0.92197/0.92319; seçilen dosyaya karşı private'ta +0.00051 / +0.00082 |
| `README.md` | v55'in kanıt tarafı temiz bir ağaçta yeniden kurulduğunda 0.66085 / 0.65164'ü birebir verdi |
| `pipeline/README.md` | hangi aşamanın ne yazdığı |

## 5. Kapatabileceğin tek açık kalem

Bu makinede base zinciri baştan koşturulup `tarik_v55_best.csv` ile byte
karşılaştırması **yapılmadı** — README bunu açıkça söylüyor. Ortamın hâlâ ayaktaysa
`run_A3.sh`'i bir kez koşturup çıkan dosyanın sha256'sını şununla karşılaştırman
yeterli:

```
41688e38733a38fed7efdb3c98662877e813e333abb9f0c3934dbdd4f2e2cadf   tarik_v55_best.csv
```

Tutarsa README'deki "tam koşu yapılmadı" uyarısını kaldırırız ve yeniden üretim
zinciri baştan sona doğrulanmış olur. Tutmazsa da bilmek isteriz — sebebi büyük
ihtimalle kütüphane sürümleridir (senin pin'lerin `requirements.txt` içinde not
düşüldü).

## 6. Yayın kararı

Depo şu an **private**. Sen onayladıktan sonra public yapılacak, sonra Kaggle'da
write-up yayımlanıp URL'si 742244 numaralı tartışmaya cevap olarak yazılacak.
**Son tarih 28 Eylül.**

- Adının geçmesini istemediğin bir yer var mı? (`LICENSE`, `README.md` künye,
  `pipeline/README.md` aşama başlıkları, `docs/TWO_CHAINS.md`)
- `docs/base_chain/EXPERIMENTS.md` 86 KB'lık tam deney günlüğün ve Türkçe; public
  olmasında sakınca var mı?
