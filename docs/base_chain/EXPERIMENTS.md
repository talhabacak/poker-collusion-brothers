# Deney günlüğü

**Bu dosya her yeni denemeden sonra güncellenir.** Her satır: ne denendi, yerel ölçüm, leaderboard sonucu, karar.
Karar sütunu: **ALINDI** (pipeline'da), **ELENDİ** (zarar verdi ya da nötr), **BEKLİYOR** (gönderilmedi).

Skor formülü: `0.70·PairAP + 0.20·EvidenceMAP@5 + 0.10·BehaviorMAP`
Ölçüm tabanı (v2, 15 Eyl): Pair **0.953**, Evidence **0.587**, Behavior **0.883**.
**Güncel ölçüm (v20 teşhisi, 17 Eyl):** 0.7·Pair + 0.1·Behavior = **0.77367**, Evidence = **0.6451**.
Behavior 0.883 varsayımıyla Pair ≈ **0.979**. Kalan ağırlıklı tavan: kanıt 0.071, çift 0.015, davranış 0.012.
**Kanıt CV'si LB'yi izliyor** (CV 0.6485 → LB 0.6451), yani kanıt kararları offline verilebilir — ama yalnız
Kural 8'e uyan yöntemlerde. Çift tarafında kullanılabilir offline ölçüt yok (Kural 9).

**En iyi gönderim: v41 = 0.90731** (18 Eyl). Zincir: MIL'siz, kolüzyon tablosuz, dördüncü aile kuralı 432 çiftte.
**Göndermeden önce:** `scratch/eda/44_predict_lb.py` ile tahmin yaz, sonra artığa bak (Kural 11).

---

## 1. Özet tablo

| # | Deney | Yerel ölçüm | Public LB | Δ | Karar |
|---|---|---|---|---|---|
| 1 | Popülasyon politika sürprizi + çift modeli (v1) | çift AP 0.965, kanıt 0.215 | 0.85842 | — | ALINDI |
| 2 | Kronolojik kanıt reranker (v2) | kanıt CV 0.470 → 0.566 | 0.87254 | +0.0141 | ALINDI |
| 3 | Metrik ayrıştırma teşhisleri (2 gönderim) | — | 0.75510 / 0.78429 | — | bilgi |
| 4 | İlişkisel LLR özellikleri (v3) | çift AP 0.9962→0.9975, beh 0.960→0.976 | 0.89100 | +0.0185 | ALINDI |
| 5 | Çift-içi stage-1b + aday 50 (v4) | kanıt CV 0.564 → 0.584 | 0.89270 | +0.0017 | ALINDI |
| 6 | PU öz-eğitim + masa özellikleri + 3 seed (v5) | yerel nötr | 0.87798 | −0.0147 | ELENDİ |
| 7 | Kısaltılmış development penceresi (TRUNC 0.67) | beh 0.978→0.954, aile doğr. 0.987→0.970 | — | — | ELENDİ |
| 8 | Aile başına stage-1b sıralayıcı (v7) | kanıt CV 0.584 → 0.611 | **0.89779** | +0.0051 | ALINDI |
| 9 | stage-1b çift özellikleri + sızıntılı reranker (v8) | sızıntılı CV 0.95 | 0.88417 | −0.0136 | ELENDİ |
| 10 | Karışık çiftler zor negatif (w=0.2) (v9) | 3 yerel ölçüt de arttı | 0.89095 | −0.0068 | ELENDİ |
| 11 | Bilgi paylaşımı dedektörü (src/07) | AUC 0.58, eval'de fazlalık yok | — | — | ELENDİ |
| 12 | Isolation özellikleri (src/03c) + 3 seed reranker (v10) | kanıt CV 0.604 → 0.620 | — | — | BEKLİYOR |
| 13 | Aile başına reranker | dürüst CV 0.618 → 0.620 (nötr) | — | — | ELENDİ |
| 14 | Epizot tarama istatistikleri | ölçüt 0.933 → 0.937 (nötr) | — | — | ELENDİ |
| 15 | Tekillik son işlemi (oyuncu başına tek ortak) | karışık ölçüt +0.055, geniş ölçüt −0.060 | — | — | ELENDİ |
| 16 | 5 kanıtı kronolojik sıralamak | MAP@5 0.618 → 0.562 | — | — | ELENDİ |
| 17 | Belirsiz ailelere `other_coordination` (v12) | belirsizlik oranı dev %2.4 vs eval %14.7 | — | — | BEKLİYOR (yedek) |
| 18 | Fold atfı: agresör null iken BB'yi agresör saymak (pipeline B fikri) | foldların %43.6'sı agresörsüz, ölçüm bekliyor | — | — | BEKLİYOR |
| 19 | Pipeline B'nin muhafazakâr other kuralı (P<0.95 ve aile el skoru<0.2) | bizde seçici değil: dev pozitiflerin 44/372'si eşik altında | — | — | ELENDİ |
| 20 | Isolation'a özel token modeli, P(iso) ağırlıklı (pipeline B'de +0.012 OOF) | — | — | — | BEKLİYOR |
| 21 | Politika modeli v2: draw'lar, betting line (src/02_policy_v2, v15) | log-loss 0.50→0.414 ama kanıt reranker 0.647→0.645, çift AP 0.9978→0.9953, behavior 0.968→0.973 | — | — | ELENDİ (nötr) |
| 22 | **Aksiyon seviyesi kolüzyon modeli** (src/04c): kanıt ellerindeki üye aksiyonları pozitif, aile başına, 5 katlama | tek başına el AUC 0.960, MAP@5 0.444; özellik olarak: stage-1b 0.517→0.535, reranker **0.620→0.647** | — | — | BEKLİYOR (v13, 17 Eyl 1. slot) |
| 24 | Zamana yerel ilişkisel taban (src/03b_relational_v2, TBIN=400, v16) | reranker 0.6454→0.6455, stage-1b 0.535→0.527, çift AP 0.9953→0.9958 | — | — | ELENDİ (nötr) |
| 25 | Isolation aksiyon dizisi token modeli (src/04d, v17) | tek başına MAP@5 0.25 (iso AUC 0.966); reranker 0.6455→0.6468 | — | — | ELENDİ (nötr) |
| 26 | Showdown board texture / loser el gücü | AUC 0.56, 0.50 (gürültü) | — | — | ELENDİ |
| 23 | BB fold atfı (src/02b): açılmamış potta agresör = big blind oyuncusu; tüm zincir yeniden (v14) | stage-1 0.488→0.468, reranker 0.647→0.645, Behavior 0.968→0.956 | — | — | ELENDİ |

---

## 2. Detaylar

### 1. Politika sürprizi (temel fikir, ALINDI)
Popülasyon aksiyon modeli (LightGBM, 2 katlama masa bazlı, 18.6M aksiyon, log-loss 0.50). Her aksiyonun sürprizi `-log p(aksiyon|durum)`. Sadece **ortak potta aktifken** toplanır. Ortak aktif değilken sinyal yok (el seviyesi AUC 0.21 vs 0.92), bu da tilt ve zayıf oyun tuzaklarını doğal olarak eler.

### 2. Kronolojik reranker (ALINDI)
Kanıt listesi çiftin ilk manipüle ellerine kayık: kanıt penceresi dışındaki yüksek skorlu eller 314 sonra / 33 önce. Aday havuzu içinde kronolojik konum en güçlü ikinci özellik. Zamansız reranker 0.480, zamanlı 0.566.

### 3. Metrik ayrıştırma (bilgi)
v2'nin kopyaları: kanıtlar boş → 0.75510, aileler `other_coordination` → 0.78429. Buradan bileşenler tam olarak çözüldü. Ağırlıkların resmi değeri sonra Kaggle MCP ile Evaluation sayfasından doğrulandı.

### 4. İlişkisel olabilirlik oranı (ALINDI)
Oyuncunun ortağa karşı aksiyon dağılımı, aynı bağlamda diğer rakiplerine karşı dağılımıyla karşılaştırılır. Bağlam: sokak × kendi el gücü × ortağa göre güç × kime karşı × heads-up (216 bağlam). Çift seviyesi G oranı tek başına AUC 0.987. Tüm veri için 47 saniye.

### 5. Çift-içi stage-1b (ALINDI)
Kanıt modelini sadece pozitif çiftler içinde, çift içinde z-skor ve yüzdelik normalize edilmiş özelliklerle eğitmek. "Hangi çift şüpheli" yerine "bu çiftin hangi eli" sorusuna odaklanır. MAP@5 0.478 → 0.500.

### 6. PU öz-eğitim (ELENDİ, −0.0147)
Etiketsiz çiftlerden skoru 0.9 üstü olanları pozitif kabul edip 2 tur yeniden eğitim. Sahte pozitif sayısı 160 → 247. Yerel etiketli AP değişmedi ama LB'de Pair AP ~0.02 düştü.

### 7. Kısaltılmış development penceresi (ELENDİ)
Evaluation'daki el sayısına benzetmek için her çiftin son %67'si. Veri azalınca aile doğruluğu ve Behavior MAP düştü, gönderilmedi.

### 8. Aile başına stage-1b (ALINDI, +0.0051)
Üç aile için ayrı sıralayıcı. Isolation 0.270 → 0.328, directed_transfer 0.570 → 0.584, soft_play 0.572 → 0.582.

### 9. Sızıntı olayı (ELENDİ, −0.0136)
Development el skorları 5 katlamanın ortalamasıyla hesaplanınca kendi katlamasını da içeriyordu. Kanıt CV'si 0.60 → 0.95 fırladı, LB düştü. **Kural: development skorları her zaman katlama dışı.** Düzeltme sonrası stage-1b'nin çift özelliği olarak katkısı 0.024'ten 0.003'e indi, yani tamamen sızıntıdanmış.

### 10. Karışık çiftler zor negatif (ELENDİ, −0.0068)
"Bir kolüzyoncu + bir masum oyuncu" çiftleri (18209 adet) 0.2 ağırlıkla negatif. Üç yerel ölçüt de arttı (etiketli AP 0.9976→0.9977, karışığa karşı 0.925→0.959, tüm etiketsize karşı 0.684→0.691) ama LB düştü. **Kural: çift modeli sadece etiketli 1860 çiftle eğitilir.**

### 11. Bilgi paylaşımı dedektörü (ELENDİ)
Ortağın kapalı kartlarını gören ikinci politika modelinin log-olabilirlik kazancı, oyuncunun diğer rakiplerine göre fazlası. Etiketli çiftlerde AUC 0.58. Evaluation'da aşırı değerli çift sayısı development'tan az. Dördüncü aile bu mekanizma değil.

### 12. Isolation özellikleri (BEKLİYOR)
Ortak aktif olma koşulu olmadan: çiftin agresyonuna fold eden üçüncü oyuncu sayısı (AUC 0.865), çok kişilik pottaki agresyon (0.825), ortağa fold (0.772). Üçü de mevcut pipeline'ın koşullu hâllerinden güçlü.

### 15. Tekillik son işlemi (ELENDİ)
`skor × sigmoid(8·(skor − oyuncunun en iyi diğer çifti))`. Karışık çift ölçütünde 0.926 → 0.980, ama tüm etiketsizlere karşı 0.676 → 0.616 ve etiketli AP 0.9975 → 0.991. Pozitiflerin 66'sında oyuncunun daha yüksek skorlu başka çifti var.

### 16. Kanıt sırası (ELENDİ)
Seçilen 5 eli kronolojik sıralamak MAP@5'i düşürdü. Reranker skor sırası iyi kalibre: pozisyon isabetleri 0.858 / 0.782 / 0.694 / 0.556 / 0.460.

### 17. `other_coordination` (BEKLİYOR)
Aile olasılığı 0.6 altında kalan çiftlerin oranı: development pozitiflerinde %2.4, evaluation yüksek riskli çiftlerinde %14.7. Entropi medyanı 0.087 vs 0.190. Açıklanmayan dördüncü ailenin evaluation'da bulunduğuna işaret. 211 çifte uygulandı.

---

### 18-21. Pipeline B'den taşınan fikirler
Arkadaşın (pipeline B) günlüğü 16 Eyl'de paylaşıldı. Kod ve veri alınmadı; yalnız ölçülmüş bulgular fikir olarak taşındı. Ayrıntı ve kural notu: `PLAN.md` §4b. Onların da bağımsız olarak doğruladığı iki kuralımız: etiketsiz çift eğitime girmez; özellik eklemek transfer eder, eğitim kümesi değişikliği etmez.

### 22. Aksiyon seviyesi model (ÖLÇÜLÜYOR)
Agresif planın ilk bahsi. Satır = çift üyesinin bir aksiyonu + ortak bağlamı (ortak aktif mi, gücü, kime karşı, katkısı, elin sonucu). Etiket = el o çiftin kanıt eli. Eğitim: ailenin pozitif çiftleri + etiketli negatif çiftlerin elleri. 10 bin pozitif satır. En önemli özellikler: sürpriz, pozisyon, ortağın pozisyonu, ortağın katkısı, oyuncu stili. Tek başına el-seviyesi modeli geçemiyor (0.444 vs 0.517) ama bağımsız bir sinyal; birleşik etki v13'te ölçülür.

### Agresif tur bilançosu (16 Eyl, 5 bahis, ~6 saat hesap)
| Bahis | Kanıt reranker CV | Sonuç |
|---|---|---|
| Aksiyon seviyesi model (v13) | 0.620 → **0.647** | tek tutan |
| BB fold atfı (v14) | 0.645, behavior −0.012 | nötr/negatif |
| Politika v2, log-loss 0.50→0.41 (v15) | 0.645 | nötr |
| Zamana yerel ilişkisel taban (v16) | 0.6455 | nötr |
| Isolation token (v17) | 0.6468 | nötr |
Çıkarım: kanıt sıralayıcısı 0.645-0.647 bandında bir platoya oturdu; bu bandı özellik ekleyerek aşmak artık pek mümkün görünmüyor. Kalan %35 kaçış muhtemelen generator'ın kanıt seçim kuralındaki bizim göremediğimiz bir bileşenden.

### 36. Çoklu örnek arıtma (MIL)
Yarışma metni "her kanıt elinde görünür, davranışa özgü **bir** aksiyon var" diyor. Model ise elin bütün aksiyonlarını pozitif sayıyordu; 11050 pozitif satırın çoğu sıradan aksiyon. İlk turun katlama dışı skorlarıyla her elde en iyi 2 aksiyon tutulup kalanı eğitimden düşürülünce (3634 satır kaldı) model belirgin keskinleşti. Bu, etiket tanımını problemin gerçek yapısına yaklaştırmak; ölçüm de onaylıyor.

| 37 | MIL sonrası "en yüksek skorlu aksiyonun tanımı" özellikleri (sokak, tür, kime karşı, sürpriz, rol) | reranker 0.6838 → 0.6866 | — | — | ALINDI |

### Kanıt bileşeninin seyri (hepsi masa bazlı OOF, LB'yi ±0.004 içinde izliyor)
| Sürüm | Kanıt CV | Not |
|---|---|---|
| v2 (kronoloji) | 0.566 | ilk reranker |
| v7 (aile başına stage-1b) | 0.611 | LB 0.89779 |
| v13 (aksiyon modeli) | 0.647 | LB katkısı +0.0029 |
| v20 (aile=model) | 0.6485 | LB 0.90268 |
| v24 (MIL 4 tur) | 0.6838 | +0.035 |
| v26 (üst-aksiyon tanımı) | **0.6866** | gönderilecek |

| 38 | Behavior boşluğunun teşhisi | az el etkisi 0.981→0.965 (küçük); %10 gizli dördüncü aile 0.970→0.862 (LB'deki 0.883 ile örtüşüyor) | — | — | bilgi |
| 39 | **Dürüst LOFO yenilik tespiti** (aileyi el skorlayıcısının eğitiminden tamamen çıkar) | görülmemiş aileyi AUC 0.966-0.981 ile ayırt ediyor; %70'ini yakalamak bilinen ailelerin %1-3'üne mal oluyor | — | — | ALINDI (src/08) |

### 38-39. Dördüncü aile: teşhis ve çözüm
Behavior MAP'in 0.883'te takılmasının sebebi ölçüldü. İki hipotez sınandı: (a) evaluation çiftlerinin az el paylaşması aile tahminini bozuyor, (b) açıklanmayan dördüncü aile her ailenin kesinliğini düşürüyor. (a) sadece 0.981 → 0.965 etkisi yapıyor; (b) %10 payda 0.970 → 0.862 yapıyor ve gözlenen değerle örtüşüyor.
Çözüm için dürüst LOFO kuruldu: bir aile el skorlayıcısının eğitiminden **tamamen** çıkarıldı (o aileye ait kafa yok), sonra o ailenin çiftleri "tanımadığım mekanizma" olarak yakalanmaya çalışıldı. Ayırt etme AUC 0.966 (isolation) ve 0.981 (soft play). Kural: çift modeli yüksek sıralıyor ama elleri kanıt modeline göre zayıf. `src/08_fourth_family.py` bu kuralı LOFO maliyet eğrisiyle kalibre ediyor.
Önemli asimetri: hedef **olmayan** bir çifte `other_coordination` yazmak faydalı (o ailenin sıralamasından yanlış pozitif çıkar); zarar sadece gerçek üç aileden birini kaçırmakta ve LOFO bunu %1-3'te tutuyor.

## 3. Çıkarılan kurallar

1. **Etiketsiz çift eğitime girmez.** İki bağımsız deneme de LB'de zarar verdi (−0.015, −0.007), ikisinde de yerel ölçütler yükselmişti.
2. **Development skorları katlama dışı olmalı.** Aksi hâlde CV şişer ve LB düşer.
3. **Yerel ölçüt seçimi kritik.** Etiketli AP doymuş (0.998). Kullanılabilir ölçütler: kanıt MAP@5 (OOF) ve "pozitifler vs bir kolüzyoncu + masum" ölçütü. Ama ikincisi eğitim kümesi değişikliklerinde yanıltıcı çıktı, sadece özellik değişikliklerinde güvenilir.
4. **Kazançlar kanıt tarafından geliyor.** Dört LB kazancının üçü kanıt modelinden (+0.014, +0.002, +0.005), biri ilişkisel özelliklerden (+0.019).
5. **Özellik eklemek transfer ediyor, eğitim kümesini değiştirmek etmiyor.**

### 26. Showdown board texture / loser strength (ELENDİ)
Her iki çiftin showdown'a gittiği ellerde (2128 el, 413 kanıt) kaybeden elin gücü ve iki el arası fark test edildi. AUC 0.56 ve 0.50, gürültü seviyesinde. Yeni bir kaldıraç değil.

### Kural 6: "skorlanan vs skorlanmayan" artefaktı
Bir el/çift skoru yalnız bir alt küme için hesaplanıp geri kalanına 0 (ya da sabit) yazılırsa, çift seviyesi ölçütlerde sahte ayrım çıkar. İki kez yaşandı: stage-1b (0.953 → düzeltince 0.927) ve aksiyon modeli (tüm etiketsizlere karşı 0.945 → sadece etiketliler skorlandığı için). **Yeni bir skor çift özelliği olarak kullanılacaksa, karşılaştırılan bütün çiftler için hesaplanmalı.**

| 27 | Aksiyon modeli toplamları çift modelinde (dürüst kapsama ile, v20) | karışık 0.9326→0.9420, tüm etiketsiz 0.6760→0.7105 | — | — | ALINDI |
| 28 | Maruziyet / uç değer normalizasyonu | karışık +0.004, birlikte kullanınca zarar | — | — | ELENDİ |
| 29 | Katılım senkronu (phi katsayısı) | tek başına AP 0.79, modele katkı yok | — | — | ELENDİ |
| 30 | Aile kaynağı = davranış modeli (kural yerine) | reranker 0.6473 → 0.6485 | — | — | ALINDI |
| 31 | v13e7: v7 riski + aksiyon modeli kanıtı | kanıt CV 0.611 → 0.647 | 0.90068 | +0.0029 | bilgi |
| 32 | **v20: aksiyon toplamları çift modelinde + aile=model kanıt** | ölçüt karışık 0.942, tüm etiketsiz 0.711 | 0.90268 | +0.0049 | ALINDI |
| 33 | v20 teşhis (kanıtsız) | — | 0.77367 | — | bilgi |
| 34 | **v21: v20 + 196 belirsiz çifte `other_coordination`** | — | **0.90387** | +0.0012 | ALINDI |
| 35 | v22: 3 seed rank-ortalamalı çift modeli | etiketli AP aynı | 0.90282 | +0.0001 | ELENDİ (gürültü) |
| 36 | Aksiyon modelinde çoklu örnek (MIL) arıtma: kanıt elinde sadece en yüksek skorlu 2 aksiyon pozitif | tur/aksiyon taraması: 2t2a 0.6547, 2t1a 0.6656, 3t1a 0.6676, 4t1a 0.6838; aksiyon modeli tek başına 0.444→0.524 | 0.90012 (v26) | −0.0026 | **ELENDİ — sızıntı, bkz. deney 50 / Kural 8** |

| 40 | Oyuncu metadata'sı (players.parquet, forumda organizatör izin verdi) | aynı bölge AUC 0.513, aynı istemci 0.508, yaş farkı 0.500; hepsi birlikte OOF AUC 0.504 | — | — | ELENDİ (sinyal yok) |
| 41 | Karşı-olgusal değer transferi (src/03d): politikanın koyacağı çip ile gerçek farkı, ortağa işaretli | kanıt 0.6866→0.6811, çift PU 0.7355→0.7143, Behavior 0.957→0.9755 | — | — | ELENDİ |
| 42 | Epizot penceresi (kayan pencereyle epizot tespiti, özellikler pencere içinden) | karışık 0.9416→0.9392, tüm etiketsiz 0.7027→0.6954 | — | — | ELENDİ |

| 43 | **Kolüzyon tablosu (AAAI 2013, Mazrooei/Archibald/Bowling)** — `src/09_collusion_table.py`: V_j değer fonksiyonu (RMSE ~21bb), C(j,k)=Σ[V_j(h·a)−V_j(h)] | çift PU ölçütü 0.7077 → **0.7438**; kanıt reranker 0.6838 → 0.6789 (zarar) | — | — | BÖLÜNMÜŞ KULLANIM |
| 44 | **Bölünmüş konfigürasyon (v33)**: kanıt zinciri CT'siz (04/04c/04b/05b), çift modeli CT'li (05) | çift PU **0.7757** (şimdiye kadarki en iyi, önceki 0.7438), kanıt reranker **0.6869** (en iyi), Behavior PU 0.7483, davranış OOF doğruluk 0.9785 | — | — | ALINDI → submission_v33 / v34_novel70 |
| 45 | Aday kümesi geri çağırma tavanı (`scratch/eda/38_cand_recall.py`) | top-50 adayda gerçek kanıt geri çağırma 0.9989, top-100'de 1.000; recall@5 = 0.627 | — | — | bilgi: NCAND darboğaz değil |

### 43-44. Kolüzyon tablosu: nereye koyulacağı önemli
Alberta'nın 2013 makalesindeki fikir, "k'nin aksiyonu j'nin beklenen değerini ne kadar değiştirdi" toplamı. Bunu el başına `help_p_by_q`, `help_q_by_p`, `ct_mutual`, `ct_asym`, `ct_max`, `ct_pos` olarak çıkardım.
Sonuç iki tarafta zıt yönde: **çift seviyesinde kuvvetli** (PU ölçütü 0.7077 → 0.7438), **kanıt seviyesinde zararlı** (0.6838 → 0.6789). Sebebi makul: değer transferi bir çiftin bütün geçmişine bakınca birikimli bir imza, tek elde ise gürültülü ve gerçek kanıt ellerini diğer yüksek-varyanslı ellerden ayırmıyor.
Bu yüzden ayrıştırıldı (`run_split.sh`): 04, 04c, 04b ve 05b `USE_CT=0`, sadece 05 `USE_CT=1`. Bu kombinasyon iki ölçütte de en iyi değeri verdi — çift PU 0.7757, kanıt 0.6869. Pair terimi metriğin %70'i olduğu için asıl umut burada.

### 45. Kanıt kaybı nerede
Aday listesi (top-50) gerçek kanıt ellerinin %99.9'unu içeriyor, yani "doğru el listeye girmiyor" diye bir sorun yok. Kayıp tamamen ilk 15 el arasında sıralama probleminde. Bu, NCAND büyütmenin (100, 200) işe yaramayacağını söylüyor — denenmesine gerek yok.

| 46 | 5 seed rank-ortalamalı çift modeli (CT'li, v33 zinciri üstünde) | çift PU 0.7757 → 0.7638, Behavior PU 0.7483 → 0.7384 | — | — | ELENDİ (deney 35 ile aynı sonuç) |

### 46. Seed ortalaması yine işe yaramadı
Deney 35'te 3 seed LB'de nötrdü (+0.0001). Yeni zincirde 5 seed ile tekrar denendi, bu sefer offline olarak da **zarar verdi**: çift PU 0.776 → 0.764. Sebep muhtemelen sadece 1.860 etiketli çiftle eğitim — seed ortalaması varyansı düşürürken tek seedin yakaladığı keskin sınırı da yumuşatıyor. Bu yön kapandı, tekrar denenmeyecek.

## 4. 18 Eylül 00:00 UTC gönderim planı (`run_submit_0918.sh`, zamanlayıcı çalışıyor)

| # | Dosya | Ne ölçüyor |
|---|---|---|
| 1 | `submission_v33.csv` | Yeni zincir, dördüncü aile kuralı yok — CT + MIL'in saf LB etkisi (referans: v21 = 0.90387) |
| 2 | `submission_v34_novel70.csv` | v33 + LOFO kuralı %70 ayar (444 çift) |
| 3 | `submission_v35_novel50.csv` | v33 + LOFO kuralı %50 ayar (309 çift) |
| 4 | koşullu | Kural eğrisi yükseliyorsa `v35_novel85` (463 çift); değilse `v26` (eski zincir kontrolü, CT katkısını izole eder) |
| 5 | rezerve | İlk dördün skoruna göre manuel |

Betik ilk üç skoru bekleyip 4. slotu kendi seçiyor, böylece bir günlük hak tek bir tahmine bağlanmıyor.

| 47 | Reranker LambdaRank kesme seviyesi (metrik MAP@5, varsayılan 10) | 2 tohum: 5 → 0.6804, 10 → 0.6832, 20 → 0.6852. 6 tohum: 10 → 0.6869, 20 → 0.6876, 35 → 0.6830 | — | — | ELENDİ (tepe yok, gürültü) |
| 48 | Dev penceresi kısaltma, çift ölçütüyle yeniden (deney 7'nin tekrarı) | TRUNC=0.68: çift PU 0.7757 → 0.7560, Behavior PU 0.7483 → 0.7386 (etiketli behavior 0.959 → 0.967 yükseldi ama PU düştü). TRUNC=0.50: 0.7172 / 0.6782 | — | — | ELENDİ (deney 7 teyit edildi) |

### 47-48. İki hiperparametre yönü de kapandı
**Kesme seviyesi:** LambdaRank'in `lambdarank_truncation_level` değeri metrikle (MAP@5) hizalı değildi. Küçültmek (5) zarar verdi, büyütmek (20) 2 tohumda +0.0020 gösterdi ama 6 tohumda +0.0007'ye indi, 35'te ise 10'un altına düştü (0.6830). Monoton bir eğilim yok, üç değer tohum gürültüsünün içinde. Varsayılan 10'da kalındı.
**Dev penceresi:** Evaluation çiftleri dev çiftlerinden az el paylaşıyor (medyan 76 vs 112, `scratch/eda/39_n_dist.py`), bu yüzden eğitimi o dağılıma çekmek mantıklı görünüyordu. Ama çift PU ölçütü düşüyor: kısaltma eldeki 372 pozitif çiftin her birini zayıflatıyor ve az veriyle eğitim kaybı, dağılım uyumu kazancından büyük. Dikkat çekici ayrıntı: etiketli behavior MAP yükseliyor (0.959 → 0.967) ama PU behavior düşüyor — tam olarak "yerel ölçüt iyileşmesi tek başına kanıt değil" dersinin tekrarı.

| 49 | **İlişkisel kolüzyon tablosu** (`src/09b_ct_relational.py`): ortak, oyuncunun diğer rakiplerinden daha mı çok yardım ediyor — oyuncu içi z-skor ve sıralama | tek başına güçlü (etiketli çiftlerde `ctr_max_hp_mean_rk` AUC **0.846** / AP 0.578, `ctr_min_hp_mean_z` 0.843 / 0.586). Çift modelinde: 12 özellik 0.7757 → 0.7594; en iyi 2 özellik 0.7723; 1 özellik 0.7617; 3 özellik 0.7566. Behavior PU 0.7483 → 0.744-0.746 | — | — | ELENDİ |

### 49. Güçlü bir özellik, yeri olmayan bir model
Ham kolüzyon tablosu "k'nin aksiyonları j'ye ne kadar yardım etti" diyor; bazı oyuncular ise zaten yapısal olarak çok yardım alır (gevşek masa, pasif rakipler). Bu yüzden ilişkisel LLR'deki normalizasyonun aynısı uygulandı: ortağın yardımı, **aynı oyuncunun diğer rakiplerinin** yardımıyla karşılaştırıldı.
Tek başına ölçüm çok iyi: AUC 0.846, sahip olduğumuz en güçlü tek özelliklerden biri (`n3_fold_to_pair` 0.865 seviyesinde). İlginç ayrıntı, ortalama yardımın işaretinin **ters** olması (AUC 0.697 ama negatif yönde): kolüzyoncu ortağından ortalamada daha az yardım alıyor, ama yardımın pozitif kısmı ve en iyi ellerdeki yoğunluğu çok yüksek. İmza "sürekli yardım" değil, "seçili ellerde yoğunlaşan yardım".
Buna karşın çift modeline hiçbir kombinasyonda katkı yapmadı — 12, 3, 2 ve 1 özellikli varyantların hepsi 0.7757'nin altında kaldı. Sebep bilgi fazlalığı: kolüzyon tablosunun kendisi (deney 43) ve ilişkisel LLR zaten çift modelinde ve aynı şeyi ölçüyorlar; 1.860 etiketli çiftle her yeni özellik varyans ekliyor.
Eşik koymamaya dikkat edildi: minimum el sayısı filtresi, filtrelenen çiftleri sabit dolgu değerinde bırakıp Kural 6'daki artefaktı yeniden üretirdi.

## 4b. 18 Eylül gönderimleri — üç bileşenin izole ölçümü

| Gönderim | Ne ekliyor | Offline vaadi | **LB** | Fark |
|---|---|---|---|---|
| v20 (referans) | — | — | 0.90268 | — |
| v21 (en iyimiz) | + dördüncü aile kuralı | — | **0.90387** | +0.0012 |
| v26 | MIL kanıt zinciri | kanıt CV 0.6485 → 0.6866 (+0.035) | 0.90012 | **−0.0026** |
| v33 | + kolüzyon tablosu (çift modeli) | çift PU 0.708 → 0.776 (+0.068) | 0.90081 | +0.0007 |
| v34 | + dördüncü aile kuralı | behavior teşhisi +0.0087'ye kadar | 0.90081 | **0.0000** |

### Kural 7 (18 Eylül akşamı DÜZELTİLDİ — aşağıdaki ilk hâli fazla genel çıktı)
**Düzeltilmiş hâli:** kanıt ölçütü (public pozitiflerde OOF MAP@5) LB'yi **iyi** tahmin ediyor — bölüm 5.1'deki
transfer tablosu ve bölüm 6'daki formül bunu gösteriyor. Bozuk olan iki özel şey var: (a) çift PU ölçütü
(Kural 9), (b) etiketi yeniden yazan yöntemlerin standart OOF'u (Kural 8). O gün üç bileşenin birden
düşmesi "bütün ölçütler bozuk" izlenimi vermişti; gerçek sebep, üçünün de bu iki kategoriden birine girmesiydi.

#### İlk (fazla genel) hâli ve o günkü gözlemler
Üç bileşen de offline kuvvetli, LB'de değersiz ya da zararlı çıktı:
- **MIL kanıt modeli**: kanıt MAP@5'te +0.035 (ağırlık 0.20 → beklenen +0.007) → gerçekleşen **−0.0026**. İşaret bile ters.
- **Kolüzyon tablosu**: çift PU ölçütünde +0.068, şimdiye kadarki en büyük yerel sıçrama → gerçekleşen +0.0007, gürültü.
- **Dördüncü aile kuralı**: v21 üzerinde +0.0012 vermişti, v33 üzerinde tam **0.0000**. Tekrarlanmıyor.

Ortak sebep hipotezi: v20'den beri her model kararı **372 public pozitif çift** üzerinde seçildi. Evaluation kümesi hem farklı çiftler içeriyor hem de açıklanmayan dördüncü aileyi (~%10). MIL gibi keskinleştirici yöntemler bilinen üç ailede iyileşirken görülmemiş ailede bozuyor olabilir — bu test edilebilir bir tahmin (LOFO ile kanıt MAP'i ölçmek).

**O günkü sonuç:** en iyi gönderim v21 = 0.90387. *(Aynı gün içinde sızıntı temizlenince v41 = 0.90731 ile
aşıldı — bkz. deney 57.)*

## 5. Offline ölçümü düzeltme (18 Eylül)

### 5.1 Kanıt ölçütü aslında çalışıyordu — MIL tek istisna
Kanıt tarafındaki her adım için offline CV farkı ile LB farkı yan yana kondu. Metrik doğrusal olduğundan LB farkı / 0.20 = LB'deki kanıt MAP@5 farkı:

| Adım | Yöntem | Offline Δ | LB Δ | İma edilen LB kanıt Δ | Transfer |
|---|---|---|---|---|---|
| v1→v2 | kronolojik reranker | +0.096 | +0.0141 | +0.071 | 74% |
| v3→v4 | stage-1b + aday 50 | +0.020 | +0.0017 | +0.009 | 43% |
| v4→v7 | aile başına stage-1b | +0.027 | +0.0051 | +0.025 | 94% |
| v7→v13e7 | aksiyon modeli kanıtı | +0.036 | +0.0029 | +0.015 | 40% |
| **v20→v26** | **MIL kanıt zinciri** | **+0.038** | **−0.0026** | **−0.013** | **−34%** |

Dört ardışık adımda offline kazanç %40-94 oranında LB'ye taşındı. Bu, kanıt CV'sinin sağlam bir ölçüt olduğunu gösteriyor. MIL tek başına işaret değiştiren istisna — bu yüzden "bütün ölçütler bozuk" değil, "MIL'de özel bir şey var" sonucu daha olası.

MIL kodunu okuyunca aday mekanizma görünüyor: r. turda f katmanının etiketleri, (r−1). turun f-modeliyle (f dışındaki katmanlarda eğitilmiş, g dahil) seçiliyor; sonra r. turun g-modeli f'nin bu seçilmiş etiketleriyle eğitilip g üzerinde değerlendiriliyor. g'nin test verisi iki adımda kendi eğitim etiketlerine sızıyor. 4 turda birikiyor. Test: `HOLDOUT_FOLD` — bir katman hiçbir tura girmez, en sonda bir kez skorlanır (`run_milhold.sh`).

### 5.2 MIL sızıntısı doğrulandı (`run_milhold.sh`, HOLDOUT_FOLD=4, 70 çift)

| | Standart OOF (iç katmanlar) | **Sıkı holdout** |
|---|---|---|
| MIL yok (1 tur) | 0.4478 | 0.4626 |
| MIL 4 tur, 1 aksiyon | **0.5177** (+0.070) | **0.4616** (−0.001) |

MIL'in +0.07'lik kazancı, hiçbir tura girmemiş katmanda **tamamen yok oluyor**. Standart 5-katlı OOF'un görmediği şey: etiket arıtması turlar arasında katmanları birbirine bağlıyor. LB'deki −0.0026 bununla açıklanıyor. Deney 36 (MIL, ALINDI) → **ELENDİ**; v26/v33/v34'ün kanıt zinciri geçersiz.

| 50 | MIL sıkı holdout testi | iç OOF +0.070, holdout −0.001 | — | — | **MIL ELENDİ, sızıntı** |

### Kural 8: Etiketleri model çıktısıyla değiştiren her yöntem (MIL, öz-eğitim, sözde etiket) standart OOF ile ölçülemez
Katmanlar etiket üzerinden birbirine bağlanır. Böyle bir yöntem yalnızca **hiçbir tura girmemiş** bir katmanda ölçülür. Bu yarışmada bu türden üç deneme yapıldı (PU öz-eğitim v5, sözde etiket, MIL); üçü de yerel ölçütte yükselip LB'de düştü. Artık bu sınıfa girecek her fikir önce holdout testinden geçer.

### Kural 9: Çift tarafında offline çözünürlük kalmadı
- Etiketli AP doymuş (0.998): fark ölçemez.
- PU ölçütü (etiketsizler negatif sayılır) yanıltıcı: 0.708 → 0.776 sıçraması LB'de +0.0007 verdi. Etiketsizlerin içinde ~1000 gizli pozitif var; ölçüt, modelin onları bulmasını **cezalandırır**, 372 açıklanmış çifti özel olarak öğrenmesini ödüllendirir.
- Tek güvenilir ölçüm: LB probu. v21'in kanıt ve davranışını sabit tutup yalnız riski değiştiren bir gönderimde LB Δ / 0.70 = Pair AP Δ, kesin. Her çift adayı 1 gönderime mal olur; kalan 12 hakla 2-3 aday ölçülebilir.

### Kanıt tarafı için geçerli protokol (çalıştığı kanıtlanmış)
Standart OOF kanıt MAP@5, Kural 8'e uyan yöntemlerde %40-94 transfer ediyor. Kanıt değişiklikleri v21'in risk/davranışı üstüne takılıp gönderilirse LB Δ / 0.20 = LB kanıt Δ; böylece her gönderim hem skor hem kalibrasyon verir.

**Geçerli en iyi zincir: v21** (0.90387). MIL'siz, CT'siz.

| 51 | **Bayat artefakt kirliliği** (`actmodel/dev_cvonly.parquet` glob'a takıldı) | reranker CV 0.6535 → 0.897, çift PU 0.721 → 0.933 — tamamen sahte | — | — | DÜZELTİLDİ + koruma eklendi |
| 52 | Temiz zincir yeniden kurulumu (MIL'siz, CT'siz = v21 konfigürasyonu) | reranker CV **0.6535**, çift PU 0.7209, Behavior PU 0.7047 | — | — | GEÇERLİ TEMEL ÇİZGİ (v40) |
| 53 | Kanıt gücüne göre geri çağırma teşhisi | rank1 0.847, rank2 0.841, rank3 0.769, rank4 0.629, **rank5 0.459**; 5 elin 3.54'ü | — | — | bilgi |
| 54 | Zorluk-dereceli lambdarank (zayıf kanıta yüksek derece) | hard 0.5431 (rank5 0.459→0.582 ama rank1 0.847→0.500), easy 0.6436, mevcut **0.6542** | — | — | ELENDİ (ödünleşim eğrisinin optimumundayız) |

### 51. Üçüncü ölçüm hatası ve kalıcı koruma
Temiz zincirin ilk çalıştırması olağanüstü sayılar verdi: reranker 0.897, çift PU 0.933. Sevinmek yerine kontrol edildi — sayılar daha önce iki kez yandığımız sızıntı imzasına benziyordu.
Sebep modelde değildi: bugünkü MIL holdout testi `actmodel/dev_cvonly.parquet` dosyasını bırakmıştı. Alt aşamalar bu klasörü `*.parquet` ile toptan okuyor, yani dev satırları iki kez yükleniyordu (223.749 → 447.498 satır, benzersiz anahtar sabit). Tek bir el top-5'te iki kez sayılınca MAP@5 kendiliğinden şişiyor.
Üç önlem alındı: (1) CV çıktısı artık glob'lanan klasörün dışına yazılıyor, (2) üç tüketicinin hepsinde `_assert_unique_keys` — glob sonrası (hidx,p,q) benzersiz değilse süreç yüksek sesle çöküyor, (3) kural olarak kaydedildi.

### Kural 10: Bir artefakt klasörü `*.parquet` ile okunuyorsa, o klasöre başka hiçbir şey yazılmaz
Yan çıktı, CV kopyası, yedek — hiçbiri. Duplikasyon sessizce metriği şişirir ve hiçbir aşamada hata vermez.

### 53-54. Kanıt kaybı zayıf manipülasyonlarda, ama oradan alınamıyor
Organizatör her çiftin beş kanıt elini güce göre sıralamış ve bizim geri çağırmamız o sıralamayı birebir takip ediyor (0.847 → 0.459). Beşinin de MAP@5'te eşit değerde olması, "kapasiteyi kolay ellere harcıyoruz" hipotezini akla getirdi.
Test bunu çürüttü. Zor ellere yüksek derece vermek gerçekten işe yarıyor (rank-5 geri çağırma 0.459 → 0.582), ama rank-1'i çok daha fazla kaybettiriyor (0.847 → 0.500) ve net MAP@5 düşüyor. Ters yön (`easy`) de düşürüyor. Mevcut ayar bu ödünleşim eğrisinin tepesinde. Zayıf manipülasyon elleri kapasite meselesi değil, gerçekten daha az ayırt edilebilir.

| 55 | Reranker konfigürasyon harmanı (full / no_rel / no_s1b / easy, çift-içi sıra ortalaması) | tekil en iyi 0.6554, en iyi harman 0.6550; sıra korelasyonu 0.948-0.988 | — | — | ELENDİ (çeşitlilik yok) |
| 56 | Küme-seçimi (5 eli bağımsız değil, birbirine zamansal yakınlık bonusuyla seçmek) | bonus 0.02 → 0.6540, 0.10 → 0.6371, 0.40 → 0.5098; kanıt tpos std 0.2345 vs tüm eller 0.2915 | — | — | ELENDİ |
| 57 | **v41: temiz zincir (MIL'siz, CT'siz) + dördüncü aile kuralı** | kanıt CV 0.6535 | **0.90731** | **+0.0034** | ALINDI — YENİ EN İYİ |

### 57. Sızıntıdan arınmış zincir gerçek kazanç getirdi
v21 (0.90387) ile v41 (0.90731) arasındaki fark +0.0034. Kanıt CV farkı yalnız +0.005 (beklenen LB etkisi +0.001) olduğuna göre kazancın bir kısmı başka yerden geliyor: 04b'nin davranış kaynağı ve dördüncü aile kuralının daha geniş uygulanması (196 → 432 çift).
Önemli olan şu: bu, üç gündür ilk kez **ileri** yönde bir adım, ve sızıntıyı temizleyerek geldi. Bir yöntemi çıkarmak, eklemekten daha çok kazandırdı.

| 58 | v43: kuralı ilk 3000 çifte genişletmek (432 → 1478 işaretli) | — | **0.90731** | **0.0000** | bilgi (çok değerli) |

### 58. Sıfır fark, büyük bilgi: pozitifler ilk 1500'de
1046 ek çifti `other_coordination` yapmak skoru **tam olarak sıfır** değiştirdi. Behavior MAP yalnız gerçek pozitif çiftler üzerinden hesaplandığına göre, 1500-3000 sıra bandında neredeyse hiç gerçek pozitif yok. Değerlendirme pozitiflerinin hemen hepsi ilk ~1500'de.
İki sonuç: (1) Pair AP'nin gerçekten yüksek olduğu (≈0.979 tahmini) bağımsız olarak doğrulandı, (2) dördüncü aile çabası yalnız ilk 1500 içinde anlamlı, ve oradaki **sayı** tek gerçek düğme. `NOVEL_KEEP` zayıf bir kaldıraçtı (0.70→0.85 yalnız 20 çift oynattı); yerine `NOVEL_N` eklendi: banttaki en "yeni" görünen N çifti doğrudan etiketler.

## 6. LB tahmin formülü (18 Eylül) — her gönderimden önce çalıştırılacak

Metrik doğrusal: `0.70·PairAP + 0.20·EvidenceMAP@5 + 0.10·BehaviorMAP`. Temiz sürümlerde ilk ve üçüncü terim
neredeyse sabit kaldığı için tek bilinmeyen kanıt terimi:

> **LB ≈ 0.7744 + 0.20 × (offline kanıt CV)**

`pair+behavior` sabiti altı temiz sürümde 0.77128-0.77661 arasında, std **0.0018**. v20'de alınan kanıtsız prob
bunu bağımsız olarak 0.77367 ölçtü — hesapla farkı 0.0008. İki ayrı yoldan aynı sayı, formül güvenilir.
Hesap: `scratch/eda/44_predict_lb.py`.

### Asıl kullanımı: artık bir teşhis aracı
| Sürüm | Tahmin | Gerçek | Artık |
|---|---|---|---|
| v41 (temiz) | 0.9051 | 0.90731 | +0.0022 (kural davranışı yükseltti) |
| **v26 (MIL)** | 0.91174 | 0.90012 | **−0.0116** |
| **v33 (MIL+CT)** | 0.91180 | 0.90081 | **−0.0110** |

Normal sapma ±0.002. Sızıntılı iki sürümde artık bunun **altı katı** ve negatif. Yani MIL gönderilmeden önce bu
hesap yapılsaydı, ilk sonucun gelmesiyle ölçütün şişkin olduğu anlaşılır ve iki gönderim daha harcanmazdı.

**Kural 11: Her gönderimden önce tahmin yaz, sonra artığa bak.** Artık −0.005'ten büyükse o yerel ölçüt bozuktur;
o yönde ikinci bir gönderim yapılmaz, önce sıkı holdout testi (Kural 8) çalıştırılır.

### Dördüncü aile eğrisi — 19 Eylül taraması bu çıkarımı ÇÜRÜTTÜ
Yukarıdaki "eğri" farklı zincirlerin karşılaştırılmasından doğan bir yanılsamaydı: v20/v21 ile v41 aynı
zincir değil, aradaki farkı kurala atfetmek hataydı. Doğrudan tarama (hepsi aynı zincir, yalnız sayı değişiyor):

| Etiketlenen çift | LB |
|---|---|
| 0 (kural kapalı) | **0.90731** |
| 150 | **0.90731** |
| 432 (v41) | **0.90731** |
| 800 | 0.89907 |
| 1200 | 0.87460 |

Kuralın katkısı **tam olarak sıfır**, fazlası zarar. 432 çifti yeniden etiketlemek skoru zerre oynatmadığına
göre o çiftlerin **hiçbiri gerçek pozitif değil** — "yenilik" skorumuz sistematik olarak pozitif olmayanları
seçiyor. 800'ü aşınca gerçek pozitiflerin ailesini bozmaya başlıyoruz.
v41'in v21'e göre +0.0034'ü tamamen **zincirden** (MIL ve CT sızıntılarının temizlenmesinden) geliyor.

## 7. 19 Eylül taraması — SONUÇ: dördüncü aile kuralı ölü

| 59 | Dördüncü aile kuralı 1-B taraması (0/150/432/800/1200 çift) | — | 0.90731 / 0.90731 / 0.90731 / 0.89907 / 0.87460 | 0.0000 / −0.008 / −0.033 | **ELENDİ — kural tamamen kaldırıldı** |

Dört gönderimlik net sonuç: kuralın katkısı sıfır, fazlası zarar. Behavior terimi 0.883'te ve bu yolla
açılmıyor. Kalan tek gerçek cep: Pair (0.979 → 1.0 = +0.0147) ve Evidence (0.645 → ? , ağırlık 0.20).

**Önemli operasyonel not:** Kaggle public LB en iyi skoru gösterir, kötü bir gönderim mevcut en iyimizi
düşürmez. Riskin maliyeti yalnızca harcanan slot; bu, yüksek varyanslı denemeleri ucuzlatıyor.


| 60 | **Dördüncü aile = davranış kafasının KARARSIZ olduğu çiftler** (`src/10_ambiguous_family.py`, 326 çift, en yüksek aile olasılığı < 0.60) | — | **0.90858** | **+0.0013** | ALINDI — YENİ EN İYİ |
| 61 | Kanıt sıralamasının tavanı | seçtiğimiz 5 elin 3.487'si doğru; kusursuz sıra 0.6540 → **0.7178** (LB'de +0.0128 değerinde) | — | — | bilgi |
| 62 | Alternatif sıralama ölçütleri (s_ev, s1b, ham sürprizal, kronoloji, harmanlar) | hiçbiri reranker'ın kendi sırasını geçmedi (0.6542 vs en iyi alternatif 0.6373) | — | — | ELENDİ |
| 63 | Kanıt seçiminde tek özellik taraması (çift-içi z, tüm özellikler) | en iyi tek özellik zaten `s_ev` (AUC 0.9332); ham sürprizal 0.9176 | — | — | bilgi: yeni sinyal yok |
| 64 | Kanıt sırası ↔ kronoloji yapısı | Spearman medyan **+0.70**, çiftlerin %49'unda r>0.8; rank 1 en erken (konum 0.227) ve en güçlü | — | — | bilgi (reranker zaten kullanıyor) |

### 60. Dördüncü aileyi doğru yerde aramak
Deney 59 eski kuralın (yüksek risk + DÜŞÜK kanıt skoru) tam olarak sıfır getirdiğini gösterdi: seçtiği 432 çiftin hiçbiri gerçek pozitif değilmiş. Yeni hipotez mekanik olarak daha sağlam: davranış kafası 3 sınıflı, açıklanmamış bir aileyi hiçbir sınıfa oturtamaz, dolayısıyla o çiftlerde **kararsız** kalır.
Asimetri de lehimize: modelin zaten emin olmadığı yerde mevcut etiketi muhtemelen yanlış, yani değiştirmenin maliyeti düşük. Sonuç +0.0013.

### 61-62. Kanıt sıralaması: büyük boşluk ama kapatılamıyor
Seçtiğimiz beş elin ortalama 3.487'si doğru. Aynı beş eli kusursuz sıralasak MAP@5 0.6540 → 0.7178 olurdu — ağırlık 0.20 ile **LB'de +0.0128**, son günlerdeki tüm kazançlarımızın toplamından büyük.
Ama denenebilir her ölçüt (s_ev, s1b, ham sürprizal, kronoloji, ağırlıklı harmanlar) reranker'ın kendi sırasının altında kaldı. Beklenen bir sonuç: skor P(kanıt)'ın monoton bir fonksiyonuysa, ona göre sıralamak beklenen AP'yi zaten maksimize eder. O boşluk modelin indirgenemez belirsizliği.

## 8. 20 Eylül — son gün planı (REVİZE: hedef 1. sıra, `run_submit_0920.sh` çalışıyor)
Kullanıcının talimatı: 1. olma ihtimali olmayan gönderim yok. 1. sıra ~0.935, biz 0.90858, fark +0.027.
Tek bileşenli hiçbir değişiklik bunu veremez (Pair tavanı +0.015, Behavior +0.010, kanıt tarafı ölçülen her
yönde tavanda). Dolayısıyla her gönderim üç yüksek-varyanslı değişikliğin **üst üste bindirilmiş** hâli:
kolüzyon tablosu çift modelinde (Pair) + davranış belirsizliği (Behavior) + denetimsiz G harmanı (Pair).

| # | Dosya | Belirsizlik eşiği | Harman w | Not |
|---|---|---|---|---|
| 1 | `s1_amb60_bl05` | 0.60 (327) | 0.05 | ılımlı paket |
| 2 | `s2_amb75_bl15` | 0.75 (596) | 0.15 | orta |
| 3 | `s3_amb85_bl30` | 0.85 (781) | 0.30 | agresif |
| 4-5 | koşullu | ilk üçün en iyisinin yönünde daha ileri (0.95 / w=0.50) | | |

Dürüst olasılık: 1. sıra için ≪%5. Ama bu, sıfır olmayan tek strateji; küçük garantili kazanç kullanıcı için değersiz.
**Kaggle final seçimi:** son sıralama özel LB'den hesaplanıyorsa, bitişten önce 2 final gönderimi elle seçilmeli
(MCP'de bu işlem yok). Aday: en yüksek public skorlu paket + v45 (güvenli).

| 65 | Harman kalibrasyonu (dev OOF risk × ilişkisel G, `scratch/eda/50_blend_calib.py`) | w=0→0.30: PU-AP 0.7365→0.7023 ama bilinen pozitiflerin üst banttaki sayısı 368→363, medyan sırası 253→245 (iyileşiyor); w=0.50'de 362, w=1.0'da 352 | — | — | bilgi: harman bilinen pozitifleri değil etiketsiz bölgeyi karıyor; w≤0.30 güvenli, 0.50 maliyet başlatıyor |
| 66 | Denetimsiz sinyalin gücü (`scratch/eda/49_unsup_strength.py`) | G_rate+G_max sıra harmanı tek başına AUC **0.9959**, PU-AP 0.6318 (model 0.7209); etiketli çiftlerde 0.9883 (model 0.998) | — | — | bilgi: harman ortağı modele yakın güçte |
| 67 | **Davranış kafası LOFO** (`BEH_LOFO=1`, 2-sınıf kafa, görülmemiş aile) | görülmemiş aileyi "kararsızlık" ile yakalama: eşik 0.75 → %8.5 (bilinen %2.6 yanlış), 0.85 → %21.8 (%4.6), 0.90 → %41.4 (%6.3), **0.95 → %77.5 (%11.4)**; soft_play görülmemişken 0.85'e kadar neredeyse hiç yakalanmıyor | — | — | bilgi: kararsızlık kuralı ancak agresif eşikte görülmemiş aileyi yakalıyor, o zaman da bilinen ailelerin %5-11'ini bozuyor |

### 65-67. Son gün paketlerinin kalibrasyonu
Üç ölçüm, üçü de offline mümkün olanın sınırında:
- **Harman** güvenli: w=0.30'a kadar bilinen pozitifler yerinde kalıyor, sadece etiketsiz bölge (gizli pozitiflerin olduğu yer) yeniden sıralanıyor. PU-AP'nin düşmesi tam da bu yüzden anlamsız (Kural 9).
- **Kararsızlık kuralı** bir madeni para: görülmemiş aile ancak 0.85-0.95 eşiğinde ciddi oranda yakalanıyor ve o eşikte bilinen ailelerin %5-11'i de "other" oluyor. Kaba hesap: 4. aile ~%10 pay ve AP'si 0'dan 0.5'e çıkarsa Behavior MAP +0.125, bilinen üç ailenin %11 kaybı ≈ −0.07 → net pozitif olabilir, ağırlık 0.10 ile ±0.01. v44'teki yenilik kuralının 800-1200 çiftte kaybetmesi de bunu doğruluyor: derinleşince gerçek pozitiflere çarpıyorsun. Fark, kararsızlığın çarptığı pozitiflerin zaten muhtemelen yanlış etiketli olması.
- Bu yüzden paket sırası doğru: 0.60 → 0.75 → 0.85 tırmanır, 0.95 yalnız 0.85 kazanırsa denenir.

## 9. 19 Eylül: takım birleşmesi — arkadaşın en büyük kazancı bizim hatta kuruldu

Arkadaşın deposu (`team-mates-repo/`) yalnız kod içeriyor, hiçbir CSV/artefakt yok ve dosya gelmeyeceği
bildirildi. Bu yüzden onun **en büyük geç kazancını** (`150_opponent_aware_policy.py`, LB'de **+0.0087/+0.0100**)
okuyup kendi hattımıza yeniden yazdım: `src/13_oppaware.py`.

**Fikir ve bizim 03b'den farkı.** Bizim ilişkisel LLR'imiz "A ortağına karşı, kendi diğer rakiplerine karşı
olduğundan farklı mı oynuyor" diye sorar — taban çizgisi oyuncunun kendisi. Bu blok farklı bir şey sorar:
**ortak elde canlıyken A'nın hamleleri, o durumlarda popülasyon politikasının beklediğinden ne kadar saptı**,
ve sapmayı modelin kendi varyansına bölerek ölçekler (`(O−ΣP)/√ΣP(1−P)`), böylece arka arkaya gelen kolay
eller sahte sinyal üretemez. Sayma-tabanlı LLR ile model-tabanlı artık farklı şekillerde yanılır; bu yüzden
ikisi birbirinin üstüne ekleniyor. Yönlü (i→j) istatistikler çift üstünde hi/lo olarak katlanıyor: G olabilirlik
oranı + χ² kuyruk derinliği, fold/call/agresyon sapmaları ve z-skorları, sürpriz ve sürpriz fazlası, ve
"A doğrudan B'nin agresyonuna cevap veriyorken" alt kümesi. 37 sütun, 12 saniye, hiçbir etiket okunmuyor.

| 68 | **Rakip-farkında politika artıkları** (`src/13_oppaware.py`) tek başına, etiketli 1860 çift | `oa_surp_excess_hi` AUC **0.9969**, `oa_g_tail_sum` AUC 0.9932 / **PU-AP 0.6853** | — | — | bilgi |
| 69 | Karşılaştırma: mevcut en güçlü özelliğimiz | ilişkisel `G_max` AUC 0.9848 / PU-AP 0.3704; tüm çift modeli PU-AP 0.7209 | — | — | bilgi |
| 70 | **OA bloğu çift modelinde, 37 sütun** | etiketli AP 0.9977 → **0.9989**, **Behavior MAP (etiketli) 0.9604 → 0.9920**, PU 0.7209 → 0.7029 | — | — | ADAY A |
| 71 | OA bloğu, en güçlü 5 sütun | etiketli AP **0.9989**, Behavior 0.9632, PU 0.7209 → **0.7351** | — | — | ADAY B |

**Okuma.** Tek bir eğitimsiz sayı (`oa_g_tail_sum`, PU-AP 0.685) neredeyse tüm eğitilmiş çift modelimiz kadar
(0.721) ayırt ediyor — mevcut en iyi özelliğimizin iki katı. 37 sütunla dürüst ölçütlerin ikisi de en iyi
değerine çıkıyor (etiketli AP 0.9989, davranış 0.9920); düşen tek şey yapısal olarak geçersiz olduğunu
bildiğimiz PU ölçütü (Kural 9) — ki gizli pozitifleri bulmayı cezalandırdığı için düşmesi kötü haber
olmayabilir. 5 sütunlu varyant PU'yu da yükseltiyor; iki aday bu yüzden kasıtlı olarak farklı yönlerde.

| 72 | **OA bloğunun el seviyesi hâli** (`oppaware_hand.parquet`, 30M satır): elde yapılan hamleler çiftin karakteristik sapmasını ne kadar örnekliyor (log tilt) | çift-içi tek özellik AUC: `oah_nll` 0.9183, `oah_tilt_hi` 0.8857 (s_ev 0.9332) | — | — | bilgi |
| 73 | **OAH sıralayıcıda** (3 tohum) | kanıt CV 0.6542 → **0.6608**; rank-4 geri çağırma 0.629 → 0.643, **rank-5 0.459 → 0.485** | — | — | ALINDI |

### 72-73. Kanıt tarafında ilk gerçek hareket
Çift seviyesindeki blok "bu ikili karakteristik olarak şu sapmayı yapıyor" diyor. El seviyesi hâli, o sapma
vektörünü (log tilt) tek tek ellere uyguluyor: bir el, çiftin fazladan ürettiği hamlelerden oluşuyorsa yüksek
skor alıyor. Kanıt terimi tam olarak bunu soruyor ve sıralayıcımızda bu bilgi yoktu.
Kazanç +0.0066 MAP ve — asıl önemlisi — **tam olarak en zorlandığımız yerde**: deney 53'te ölçtüğümüz
"zayıf manipülasyon ellerini kaçırıyoruz" sorunu ilk kez kıpırdadı (rank-5: 0.459 → 0.485). Deney 54'te
zorluk-ağırlıklandırmanın yapamadığı şeyi yeni bilgi yaptı.

### Metrik gerçeği (19 Eylül akşamı, hedef 0.938)
| Senaryo | Lideri yakalamak için gereken kanıt MAP |
|---|---|
| Pair 1.00, davranış 1.00 | 0.690 |
| Pair 0.99, davranış 0.98 | 0.735 |
| Pair 0.98, davranış 0.97 | 0.771 |

Biz kanıtta 0.645'teyiz. **Pair ve davranış mükemmel olsa bile kanıt 0.645'te kalırsa tavan 0.929.**
Dolayısıyla kalan tüm çaba kanıt terimine gitmeli; OA bloğunun Pair/davranış kazancı değerli ama eşiği
tek başına geçirmez.

| 74 | Bahis boyutu imzası sıralayıcıda (`src/14_sizing_hand.py`, 18 sütun) | kanıt CV 0.6608 → **0.6588** (3 tohum, aynı taban) | — | — | ELENDİ |

### 74. Boyut bloğu bizde transfer etmedi
Arkadaşın ablasyonunda boyut bloğu kronolojiden sonra en değerli ikinci bloktu (−0.0244 MAP çıkarılınca).
Bizde tersine küçük bir düşüş verdi. Muhtemel sebep: bizim aksiyon modelimiz `size_z`'yi zaten aksiyon
seviyesinde kullanıyor ve el seviyesi özetleri yeni bilgi değil, aynı bilginin gürültülü tekrarı oluyor.
Onun hattında aksiyon modeli yok, boşluğu boyut bloğu dolduruyordu. **Ders: başka bir hattın ablasyon
değeri, kendi hattında aynı boşluk yoksa transfer etmiyor.**

## 10. 20 Eylül gönderim adayları (hazır, doğrulandı)

| Dosya | Kanıt CV | İçerik |
|---|---|---|
| `submission_A2_oa_both.csv` | **0.6609** | OA çift + OA el + aile belirsizlik kuralı (495 çift) |
| `submission_v54.csv` | 0.6609 | aynısı, kural kapalı |
| `submission_A_oa_full.csv` | 0.6535 | OA çift + eski kanıt zinciri + kural |
| (referans) v45 = 0.90858 | 0.6535 | OA yok |

| 75 | **Rakip-koşullu politika** (`src/15_oppcond_policy.py`): beklentinin kendisi rakibin kimliği, pozisyonu ve altı stil oranıyla koşullandırılıyor; 17.1M (aksiyon × canlı rakip) satırı, tutturma hatası 0.469/0.473 | tek başına PU-AP **0.6744** (adım 13: 0.6853) — daha zayıf | — | — | tek başına ELENDİ |
| 76 | **Adım 13 + adım 15 birlikte, sıralayıcıda** | kanıt CV 0.6608 → **0.6670** (3 tohum); rank-3 0.763→0.780, rank-5 0.485→**0.506** | — | — | ALINDI |
| 77 | Her iki blok, tam zincir (6 tohum) | kanıt CV **0.6658**, etiketli AP **0.9989**, Behavior **0.9920** | — | — | ADAY A3 |

### 75-77. Keskin model paradoksu, ve neden ikisi birden gerekiyor
Rakip-koşullu beklenti mantıken daha doğru: "bu rakip diğerlerine benzemiyor" meşru bir sebep ve onu
ayıklayınca geriye saf sinyal kalmalı. Ama tek başına **daha zayıf** çıktı (0.6744 vs 0.6853) — model
keskinleştikçe kolüzyon hamlelerine de az şaşırıyor. Bu, deney 3'te politika v2'yi (log-loss 0.500→0.414,
kanıt nötr) eleyen aynı mekanizma.
Buna rağmen **üst üste konunca kazandırıyor**: 0.6608 → 0.6670. İki blok aynı şeyin iki farklı okuması
değil; biri "durumun beklediğinden saptı", diğeri "rakibin de hesaba katıldığı beklentiden saptı" diyor ve
aradaki fark bilgi taşıyor. Ders: tek başına zayıf bir sinyal, koşullandırması farklıysa yine de eklenir.

## 11. 20 Eylül adayları — nihai (hiçbiri gönderilmedi, kullanıcı onayı bekleniyor)

| Dosya | Kanıt CV | Etiketli AP | Behavior | İçerik |
|---|---|---|---|---|
| `submission_A3_both_blocks.csv` | **0.6658** | **0.9989** | 0.9920 | İki blok + belirsizlik kuralı (481 çift) |
| `submission_v55.csv` | 0.6658 | 0.9989 | 0.9920 | Aynısı, kural kapalı |
| `submission_A2_oa_both.csv` | 0.6609 | 0.9988 | 0.9922 | Tek blok + kural |
| (mevcut en iyi, gönderilmiş) v45 | 0.6535 | 0.9977 | 0.9604 | Blok yok — **LB 0.90858** |

Takımlar birleşti: gönderim kotası ortak, listede arkadaşın v58-v70 dosyaları da görünüyor, birleşik en iyi 0.90858.

| 78 | **v55: her iki rakip-farkında blok, çift modeli + kanıt sıralayıcısı, dördüncü aile kuralı yok** | kanıt CV 0.6658, etiketli AP 0.9989, Behavior 0.9920 | **0.91748** | **+0.0089** | ALINDI — YENİ EN İYİ |

### 78. Kazanç nereden geldi: tahmin formülünün artığı söylüyor
Tahmin `0.7744 + 0.20 × 0.6658 = 0.90756`, gerçek **0.91748**, artık **+0.0099** — normal sapmanın (±0.0018) beş katı ve **pozitif**. Formülün sabiti `0.7P + 0.1B` terimini varsaydığı için, bu artık doğrudan o terimin yükseldiği anlamına geliyor: 0.7743 → **0.7843**.
Ayrışma: kanıt tarafı +0.0025 (CV +0.0123 × 0.20), çift+davranış tarafı **+0.0100**. Bu, arkadaşın kendi hattında aynı bloktan ölçtüğü +0.0087/+0.0100 ile birebir örtüşüyor — yöntem iki bağımsız hatta aynı büyüklükte transfer etti.
**Not:** artığın pozitif ve büyük olması Kural 11'in ters yönü; formülün sabiti artık geçersiz, çünkü o sabit "çift ve davranış sabit kalır" varsayımına dayanıyordu ve bu sürüm tam olarak o varsayımı bozdu. Yeni sabit: **0.7843**.

## 12. Güncel durum (20 Eylül 00:15 UTC)
| | |
|---|---|
| En iyi skor | **0.91748** (v55) |
| Önceki en iyi | 0.90858 (v45) |
| Birleşik takım, kalan hak | 4 (bugün, ortak kota) |
| Bitiş | 20 Eylül 22:00 UTC |

## 13. 20 Eylül: yeni yöntem araştırması (kullanıcı: "risk al, korkak davranma")

Üç yeni eksen açıldı. İlki kendi fikrimiz, diğer ikisi arkadaşın deposunun tam envanterinden çıktı
(o envanter aynı zamanda birkaç ekseni **ölçülmüş olarak** kapattı — aşağıda "kapalı" olarak işaretli).

### Kolüzyon halkaları var mı? (kendi fikrimiz)
Etiketli 372 pozitif çiftte 49 oyuncu birden fazla pozitif çiftte görünüyor; çift grafiğini sabit tutan
permütasyon null'u 25.4 ± 4.2 diyor → **5.6 sigma**, yani kolüzyon bir oyuncu özelliği de. Kapalı üçgen
sayısı ise **sıfır**: yapı yıldız biçiminde, halka değil.

| 79 | **Çift-grafiği özellikleri** (`src/16_graph_pair.py`, 28 sütun: hub = oyuncunun *diğer* en iyi ortağı, dışlayıcılık = oyuncunun kendi ortak dağılımı içindeki sıra/z/pay, üçgen desteği) | tek başına: dışlayıcılık sütunları tabanın yeniden ifadesi (AUC 0.9928 vs taban 0.9932), hub/üçgen tek başına sıfır. Modelde: etiketli OOF AP 0.99856 → 0.99842 (+graf) / 0.99857 (yalnız hub); evrende 372 bilinen pozitifin **hepsi zaten ilk 1000'de**, medyan sıra 278 → 271 | — | — | ZAYIF POZİTİF, raftа |

**Okuma.** Hipotez doğru ama kullanılabilir kazanç yok: en zor 40 pozitifin %42.5'i hub bağlantılı (taban
%25.8), yani yapı tam da zorlandığımız yerde yoğunlaşıyor — ama model bilgiyi zaten başka yoldan içeriyor.
Arkadaşın deposu aynı sonuca iki kez daha varmış (`80_player_graph_features.py` ELENDİ; `300_relational_graph_residual.py`
"görünür kazancın tamamı sızıntı, dosya yazılmadı"). **Eksen kapatıldı.**

### Arkadaşın deposunun tam envanteri — bizde karşılığı olmayanlar
Onların kendi ablasyonunda kanıt motorunu en çok taşıyan blok **epizot/konum (11 sütun): −0.1458 MAP@5**,
yani ELEDİĞİMİZ bahis-boyutu bloğunun (−0.0244) **altı katı**. Bizde karşılığı yoktu.

| 80 | **Temas-kısıtlı kronoloji + komşuluk + duvar saati** (`src/17_chrono_hand.py`, 17 sütun) | çift ~120 el paylaşıyor ama gerçek temas medyan **42** elde; konum artık o alt küme içinde ölçülüyor. Komşu el (kendisi hariç) transfer/sürpriz ortalaması, 9-el yoğunluğu, önceki/sonraki ele saniye farkı (medyan 77 sn, uzun kuyruk = oturum kesintisi) | — | — | ölçülüyor |
| 81 | **Bırak-birini-dışarıda yön tutarlılığı** (`src/05b`, `USE_DIR=1`, 7 sütun) | bu el, çiftin *diğer* ellerinin taşıdığı yönde mi değer taşıyor (el kendine oy vermiyor); artı preflop equity farkı aynı yönde işaretlenmiş — güçlü eli olanın potu bırakması | — | — | ölçülüyor |
| 82 | **Çift-içi normalizasyon** (`RERANK_PAIRNORM=1`): yeni rakip-farkında sütunların hepsi için çift içi sıra + z | ölçek çiftin özelliği (kaç el, ne kadar gevşek), elin değil; 1860 grupla modelin bunu ham değerlerden öğrenmesi zor | — | — | ölçülüyor |
| 83 | **Yumuşak yönlendirilmiş uzmanlar** (`RERANK_ROUTED=1`): havuz modeli + üç aile uzmanı, çift-içi sıralar üzerinden `0.25·havuz + 0.75·Σ_f P(aile=f)·uzman_f`. Bizim `RERANK_PER_FAMILY` ya/ya da idi; onlarda bu mimari +0.0091 | — | — | — | ölçülüyor |

### Envanterin kapattığı eksenler (tekrar denemeyeceğiz)
- **Sinir ağı / gömme / dizi modelleri:** dördü de denenmiş. Dikkat-tabanlı MIL sıralayıcı GBDT'ye **0.0301 kaybetmiş**, beş holdout katının hepsinde negatif. Uçtan uca öğrenilen çift latenti (49M satır) gerçek — permütasyon null'unu 2.8× aşıyor — ama `oa_*` toplamlarının üstüne **koşullu katkısı 25 basamağın hiçbirinde yok**. JEPA/transformer yolu ölçülerek kapanmış.
- **Epizodiklik (kümelenme):** zaman-karıştırılmış kontrolde kazanç sağ kalıyor → ölçülen şey yoğunluk, epizodiklik değil. Üstelik yerleştirilmiş eller zaman çizgisinin %55.8'ine yayılmış.
- **Graf/halka:** yukarıda.

### 80-83 sonuçları: yedi eksenin altısı kapandı, biri açıldı
Kanıt CV tabanı 0.6670 (v55 zinciri, 3 tohum, RERANK_CV_ONLY ile ölçüldü; 0.66696 birebir yeniden üretildi).

| Deney | Yöntem | Kanıt CV | Karar |
|---|---|---|---|
| 80 | Temas-kısıtlı kronoloji + komşuluk + saat boşlukları (`src/17_chrono_hand.py`) | **0.6650** | tek başına ELENDİ |
| 82 | Çift-içi normalizasyon (`RERANK_PAIRNORM=1`) | **0.6649** | ELENDİ |
| 80+82 | İkisi birlikte | **0.6618** | ELENDİ |
| 81 | Bırak-birini-dışarıda yön tutarlılığı (`USE_DIR=1`) | **0.6621** | ELENDİ |
| 83 | Yumuşak yönlendirilmiş uzmanlar (`RERANK_ROUTED=1`) tek başına | **0.6674** | nötr |
| 83+80 | **Uzmanlar + kronoloji** | **0.6710** | **ALINDI (+0.0040)** |

| 84 | **İlk 5'i yeniden sıralama** (aynı beş el, 24 kural: s_ev, erkenlik, sürpriz ve s2 ile harmanları) | mevcut sıra 0.6543, en iyi alternatif 0.6545 | — | — | ELENDİ (deney 62 teyit) |
| 85 | **Faz karşıtlığı** (`/tmp/phase.py`): aynı çiftin diğer fazdaki değeri, çifte özgü ücretsiz null | mekanizma **doğrulandı** — dev-pozitiflerinin eval-penceresi AUC'si 0.477-0.504, yani saf şans; kolüzyon faza özgü. Ama modelde: etiketli AP 0.99860 → 0.99859, evren medyanı 276 → 273 | — | — | ELENDİ (bilgi zaten içeride) |

### Neden çoğu ELENDİ — ve tek istisnanın anlamı
Altı eksenin ikisinde hipotez **doğrulandı** (oyuncu düzeyi yapı 5.6σ; faz özgüllüğü AUC 0.48) ama hiçbiri
kazanca dönüşmedi. Tekrarlayan desen şu: bizim zincirimiz (aksiyon modeli + stage1b + rakip-farkında iki blok)
bu bilgiyi zaten başka bir yoldan taşıyor, yeni sütun aynı bilginin gürültülü tekrarı oluyor. Deney 74'ün
dersinin genellemesi: **başka bir hattın ablasyon değeri, kendi hattında aynı boşluk yoksa transfer etmiyor.**

Tek istisna öğretici: kronoloji **tek başına zarar verdi (−0.0020) ama aile uzmanlarıyla birlikte kazandırdı
(+0.0040)**. Zamanlama yapısı aileye özgü — yumuşak play'in epizodu ile yönlendirilmiş transferin epizodu
farklı biçimde yerleşiyor — ve havuz modeli üç aileyi tek kalıba sıkıştırınca sinyal birbirini götürüyor.
Mimari değişikliği olmadan özellik işe yaramıyor; özellik olmadan mimari değişikliği nötr (0.6674).

## 14. 20 Eylül: uzman mimarisi elenen blokları geri açıyor

Deney 83+80'in bulgusu (kronoloji tek başına −0.0020, uzmanlarla +0.0040) tesadüf değilmiş. Aynı desen
üç blokta birden tekrarladı: **havuz modeli üç aileyi tek kalıba sıkıştırırken sinyaller birbirini
götürüyor; aile uzmanlarına bölünce her biri kendi ailesinin imzasını kullanabiliyor.**

| # | Yapılandırma (hepsi 3 tohum, taban 0.66696) | Kanıt CV |
|---|---|---|
| 86 | Uzman ağırlığı taraması: 0.50 / 0.75 / 0.90 / 1.00 | 0.6681 / **0.6710** / 0.6693 / 0.6671 — tepe 0.75'te (arkadaşın bulduğu değer) |
| 87 | Sıralayıcı kapasitesi (hiç taranmamıştı): 7 yaprak / 31 yaprak / 800 tur lr.015 / min_child 10 | 0.6653 / 0.6709 / 0.6690 / 0.6706 — mevcut 15 yaprak/400 tur zaten tepe |
| 88 | Uzman + kronoloji + **çift-içi normalizasyon** | **0.6716** |
| 89 | + **aileye özel parametreler** (dt 7 yaprak/300 tur, sp 31/500, iso 7/300) | 0.6712 (tek başına nötr) |
| 90 | + boyut bloğu (deney 74'te elenmişti) | 0.6685; **aile parametreleriyle birlikte 0.6723** |
| 91 | + **yön tutarlılığı** (deney 81'de elenmişti) + aile parametreleri | **0.6785** |
| 92 | + değer bloğu | 0.6646 (elendi) |

**Deney 91, taban üstüne +0.0115.** Üç ayrı ELENDİ kararı (kronoloji −0.0020, yön −0.0049, aile parametreleri
nötr) birlikte kullanıldığında pozitife dönüyor. Bu, Kural 12'yi doğuruyor.

> **Kural 12.** Bir bloğun tek başına ölçülen değeri, modelin o bloğu kullanabilecek kapasitesi yoksa
> anlamsızdır. Aynı özelliği üç aile için tek fonksiyona sıkıştıran bir sıralayıcıda aileye özgü sinyaller
> birbirini götürür. Blok elemeden önce mimarinin o bloğu ifade edebildiğinden emin ol.

| 93 | Yumuşatılmış uzmanlar (`FAM_SMOOTH`): uzman kendi ailesi dışındakileri düşük ağırlıkla da görsün (3× veri) | 0.15 → 0.6732, 0.30 → 0.6718, 0.50 → 0.6700 (referans 0.6748) | — | — | ELENDİ, monoton düşüyor — sert uzman doğru |
| 94 | **EN İYİ YAPILANDIRMA, 6 tohum**: `RERANK_ROUTED=1 USE_CH=1 RERANK_PAIRNORM=1 USE_DIR=1 FAM_TUNE=1` | kanıt CV **0.6748** (üretim tabanı 0.6658, **+0.0090**); rank-5 geri çağırma 0.506 → **0.5265** | — | — | **ADAY v56** |

### 94. Tohum gürültüsü uyarısı
3 tohumla 0.6785 okunan yapılandırma 6 tohumla 0.6748'e, başka bir 3'lü tohum kümesiyle 0.6712'ye indi.
~30 yapılandırma tarandıktan sonra en yüksek okuma sistematik olarak şişkin. **Dürüst rakam 6 tohumluk
0.6748'dir** ve LB karşılığı `0.20 × 0.0090 = +0.0018` → beklenen ~0.9193.

Bileşen ablasyonu (3 tohum, en iyinin 0.6785'i üstünden): aile parametreleri çıkınca 0.6732,
kronoloji çıkınca 0.6735, çift-içi normalizasyon çıkınca 0.6697. Üçü de gerekli; hiçbiri tek başına değerli değil.

### v56 dosyası
`run_v56.sh`. Risk ve davranış sütunları v55 ile **birebir aynı** (değişiklik yalnız kanıt tarafında,
temiz izolasyon). İlk kanıt elinin %74.7'si aynı, ilk-5 küme örtüşmesi 4.47/5. `06_validate_submission` OK.
Not: değerlendirme geçişi artık çiftleri kovalara bölerek çalışıyor (`EVAL_BUCKETS`, varsayılan 6) —
yeni ~70 sütunla tek parça geçiş belleğe sığmıyor ve OOM ile öldürülüyordu.

### 95-97. Üç tohum kümesiyle disiplinli doğrulama
~30 yapılandırma taradıktan sonra tek bir tohum kümesindeki en yüksek okuma güvenilmez. Bundan sonra her
karar **üç bağımsız 6-tohumluk kümede** (A=42,7,2024,11,99,5 · B=3,13,23,33,43,53 · C=101,202,303,404,505,606)
ölçüldü ve ortalaması alındı.

| 95 | **Aileye özel kapasite ayarı** (DT/SP/CI için yaprak ve tur taraması, 16 yapılandırma) | tohum A'da en iyi kombinasyon 0.6805 (+0.0058) ama **bağımsız tohum kümesi B'de 0.6746 vs referans 0.6743 = +0.0003** | — | — | **ELENDİ — kazancın tamamı seçim gürültüsü** |
| 96 | **v56 adayı vs v55 tabanı, üç kümede** | taban 0.6658 / 0.6624 / 0.6610 (ort **0.6631**) → aday 0.6748 / 0.6743 / 0.6730 (ort **0.6740**), **üç kümede de +0.009 ila +0.012** | — | — | **GERÇEK, +0.0110** |
| 97 | **Davranış kafasının kalibre olasılıklarıyla yönlendirme** (`BEH_ROUTE=1`): yönlendirici artık aşama-1 aile skoru payları değil, çift modelinin OOF softmax'ı | 0.6770 / 0.6766 / 0.6738 (ort **0.6758**) — **üç kümede de pozitif**, +0.0018 | — | — | ALINDI |
| 98 | Uzman ağırlığı yeni yönlendiriciyle yeniden: 0.75 / 0.85 / 0.90 / 1.00 | ort 0.6758 / 0.6764 / **0.6768** / 0.6771 — düz; 1.00 havuz modelini tamamen atıyor, en kötü kümede (C) 0.90 daha iyi | — | — | **SPEC_W=0.90 seçildi** |

### Nihai: v57
`run_v57.sh` — `RERANK_ROUTED=1 USE_CH=1 RERANK_PAIRNORM=1 USE_DIR=1 FAM_TUNE=1 BEH_ROUTE=1 SPEC_W=0.90`, 6 tohum.
Üç kümenin ortalaması **0.6768**, v55 tabanı 0.6631 → **+0.0137 kanıt CV** → LB'de `0.20 × 0.0137 = +0.0027`
→ beklenen **~0.9202**. Risk ve davranış sütunları v55 ile birebir aynı; değişiklik yalnız kanıt tarafında.

> **Kural 13.** Bir yapılandırma taramasından çıkan en iyi okuma, taramada kullanılmayan bir tohum kümesinde
> tekrarlanmadıkça gerçek değildir. Deney 95'te +0.0058 okunan ayar bağımsız kümede +0.0003 çıktı; deney 96'da
> +0.0110 okunan değişiklik üç kümenin üçünde de tekrarlandı. Fark, kararın tamamı.

| 99 | **LambdaRank kesme seviyesi, yönlendirilmiş mimaride yeniden** (deney 47 havuz mimarisinde tepe bulamamıştı) | üç kümenin ortalaması: kesme 5 → 0.6722, 10 → 0.6768, 15 → 0.6786, 20 → 0.6796, 30 → 0.6798, **50 → 0.6812** | — | — | **ALINDI, kesme 50 (= hiç kesmeden)** |
| 100 | Tohum sayısı: 18 tohum tek koşu vs üç 6-tohumluk koşunun ortalaması | 0.6794 vs 0.6796 — aynı şey; ayrıca kesme 50'de tohum yayılımı 0.6808-0.6815'e daralıyor | — | — | bilgi: 6-12 tohum yeterli |
| 101 | 100 aday + kesme 100 | ort 0.6812 (0.6828/0.6799/0.6811) — kazanç yok, yayılım daha geniş | — | — | ELENDİ, NCAND=50 kalıyor |
| 102 | Kesme 50'de uzman ağırlığı 1.0 (havuz modeli tamamen atılıyor) | ort 0.6805 vs 0.90'ın 0.6812 | — | — | ELENDİ, SPEC_W=0.90 kalıyor |

### 99. Kesme seviyesi neden şimdi önemli hâle geldi
Deney 47 aynı parametreyi havuz mimarisinde taramış ve düz bulmuştu (10 → 0.6869, 20 → 0.6876, 35 → 0.6830).
Yönlendirilmiş mimaride ise monoton ve güçlü: 0.6722 → 0.6812. Sebebi Kural 12'nin aynısı — kesme seviyesi
modelin listenin ne kadar derinini öğrenmeye çalıştığını belirler; üç aileyi tek fonksiyona sıkıştıran bir
model derin listeyi zaten öğrenemediği için parametre etkisiz kalıyordu. Uzmanlara bölününce her biri kendi
ailesinin tam sıralamasını öğrenebiliyor ve tam liste (kesme 50 = NCAND) en iyisi oluyor.
**Aynı parametre, aynı veri, farklı mimari — biri düz, diğeri +0.0090.**

## 15. Nihai aday: v58
`run_v58.sh`: `RERANK_ROUTED=1 USE_CH=1 RERANK_PAIRNORM=1 USE_DIR=1 FAM_TUNE=1 BEH_ROUTE=1 SPEC_W=0.90 LR_TRUNC=50`, 12 tohum.

| | kanıt CV (üç kümenin ort.) | LB |
|---|---|---|
| v45 (rakip-farkında blok yok) | 0.6535 | 0.90858 |
| v55 (gönderildi) | **0.6631** | **0.91748** |
| v56 | 0.6740 | — |
| v57 | 0.6768 | — |
| **v58** | **0.6812** | tahmin **~0.9211** |

v55 → v58 farkı **+0.0181 kanıt CV**, LB karşılığı `0.20 × 0.0181 = +0.0036`.
Risk ve davranış sütunları v55 ile birebir aynı; tüm değişiklik kanıt tarafında, yani tahmin formülünün
sabiti (0.7843) bu kez geçerli kalıyor ve tahmin sıkı olmalı.

| 103 | **Yönlendirilmiş risk modeli** (`RISK_ROUTE_TEST=1`, `src/05_pair_model.py`): kanıt tarafındaki uzman fikri metriğin %70'ini taşıyan çift modeline uygulandı — aynı negatifler, pozitifleri tek aileye kısıtlanmış üç uzman, davranış OOF olasılıklarıyla harmanlanıyor | 5 tohum: ilk200 143.2 → **147.8** (5/5 tohumda pozitif), medyan 256.4 → 254.2, ama **ilk500 362.6 → 360.6** ve etiketli AP 0.99884 → 0.99873 (5/5'te düşük) | — | — | **ELENDİ** |
| 104 | Çift modelinde tohum ortalaması (koşular arası kararsızlığa karşı) | ilk200 144 vs tek tohum ortalaması 143.2 — fark yok (deney 46 teyit edildi) | — | — | ELENDİ |

### 103. Neden kanıt tarafında işe yarayan şey çift tarafında yaramadı
Uzman mimarisi kanıt sıralayıcısında +0.018 getirdi, çift modelinde ise yönü belirsiz: en tepede (ilk 200)
tutarlı biçimde iyi, biraz aşağıda (ilk 500) tutarlı biçimde kötü, etiketli AP'de beş tohumun beşinde de
hafif düşük. Fark, görevin doğasında: **kanıt seçimi aileye göre gerçekten farklı bir görev** (soft play'in
epizodu ile yönlendirilmiş transferin epizodu farklı yerde ve farklı biçimde), **risk skorlaması ise değil** —
üç ailenin hepsinde aynı soru soruluyor: bu ikili anormal mi. Uzmanlara bölmek burada yalnızca her modelin
gördüğü pozitif sayısını üçe bölüyor.
Metriğin %70'ini taşıyan sütunu, yönü belirsiz ve tek güvenilir ölçütü doymuş bir sinyal için değiştirmek
kötü bahis. **Çift tarafı v55'teki hâliyle kalıyor.**

### Not: çift modeli koşular arası tam belirleyici değil
Aynı komutla iki kez çalıştırılan `05_pair_model.py` farklı risk sütunu üretiyor (en büyük fark 0.084,
davranış tahminlerinin %98'i aynı). Bu yüzden `submission_v55_base.csv` **asla yeniden üretilmemeli**;
v56/v57/v58 hep o orijinal dosyanın üstüne kuruldu, böylece risk ve davranış sütunları LB'de ölçülmüş
0.91748'in tam olarak aynısı.

## 16. Arkadaşın strateji dokümanları (`docs/`, 22 md) — okundu ve ölçüldü

### Dördüncü aile savunması: onlarda üretimde, bizde gereksiz
`OTHER_COORDINATION_REVIEW.md` üretimde bir **risk tabanı** taşıyor: `max(tail(model), tail(ps_top5) − 0.5)`.
Gerekçesi ölçülmüş: bir aileyi hiç görmemiş çift modellerinin o ailedeki AP'si **0.958 → 0.113**,
0.916 → 0.031, 0.960 → 0.073 diye çöküyor; eğitimsiz `ps_top5` ise 0.833 / 0.503 / 0.791 ile buluyor.
Bizde aynı istatistik `surp_top5` olarak zaten var.

| 105 | **Görülmemiş aile dayanıklılığı, bizim hatta** (`GUARD_TEST=1`; sızıntısız 82 sütun: yalnız `oa_*`, `oc_*`, `G_*`, `rel_*`, `surp*`) | saklanan ailede model AP: directed **0.5495**, soft play **0.3172**, izolasyon **0.2689** — arkadaşın 0.113/0.031/0.073'ünün ~5 katı | — | — | bilgi: **zaten dayanıklıyız** |
| 106 | Risk tabanı (`max(pct(model), pct(surp_top5) − c)`), c=0.50 / 0.25 | tam olarak hiçbir şey değiştirmiyor — model görülmemiş aile pozitiflerini zaten tabanın üstünde sıralıyor | — | — | ELENDİ (işlevsiz) |
| 107 | **Eğitimsiz istatistikler saklanan ailede modeli geçiyor** | directed: `surp_top5` 0.665 (model 0.549); soft play: `oa_g_tail_sum` **0.705** (model 0.317); izolasyon: `surpmax_top5` **0.721** (model 0.269) — her ailede farklı bir istatistik | — | — | bilgi (çok değerli) |
| 108 | Beş eğitimsiz istatistiğin komitesi (yüzdelik sıra ortalaması), harman ağırlığı w | ayrı ölçeklerde hesaplanan "beklenen PairAP" w=0.1'de +0.0025/+0.0054/+0.0109 diyordu | — | — | **YANILTICI — protokol hatası** |
| 109 | **Karışık AP, dürüst protokol** (`MIXAP_TEST=1`): gizli aile ve bilinen aileler **tek sıralamada**; risk modeli saklanan aile olmadan çapraz doğrulanıyor, gizli aile pozitifleri f payına indiriliyor | f=0.05: w=0 **0.6726**, w=0.05 −0.0051, w=0.1 −0.0119, w=0.2 −0.0217 · f=0.10: −0.0032 / −0.0091 / −0.0176 · f=0.20: −0.0011 / −0.0050 / −0.0108 | — | — | **ELENDİ — her f değerinde zarar** |

### 108 → 109: ölçekleri toplamak sahte kazanç üretiyor
Deney 108'in "beklenen PairAP"si iki farklı popülasyona karşı hesaplanmış iki AP'yi ağırlıklı topluyordu:
bilinen aile AP'si 1488 kolay negatife karşı, görülmemiş aile AP'si 155 bin etiketsize karşı. Aynı ölçekte
değiller. Tek sıralamada dürüstçe ölçülünce işaret **tersine dönüyor**. Bu, Kural 9'un yeni bir yüzü.

> **Kural 14.** İki ayrı AP'yi ağırlıklı toplayarak "beklenen metrik" üretme. Metrik tek bir sıralama
> üzerinden tanımlıysa, simülasyon da tek bir sıralama üzerinde yapılmalı — popülasyonlar birlikte sıralanmalı.

### Resmî puanlayıcı uç durum denetimi (`01_KOD_VE_OLCUM_DENETIMI.md` §A6)
Resmî puanlayıcı yerel olandan dört uç durumda ayrılıyor; en tehlikelisi: **aynı çiftte tekrarlanan kanıt eli
= gönderim hatası**. `submission_v55 / v58 / v58_amb` üçü de temiz: tekrarlı kanıt eli 0, `NO_EVIDENCE` 0,
üç aile de dolu, risk sonlu ve (0,1) içinde, 112.540 tekil çift, null yok.

## 17. İzolasyon ailesine nişan alma — ölçüldü, kapandı

Kanıt tarafında aile bazında nerede olduğumuzu ilk kez ölçtük (`map5_by_family`, 6 tohum, tohum kümesi A):

| Aile | çift | v55 | v58 | fark |
|---|---|---|---|---|
| directed_transfer | 148 | 0.7004 | **0.7131** | +0.0127 |
| soft_play | 132 | 0.6483 | **0.6713** | +0.0230 |
| **coordinated_isolation** | 92 | 0.6353 | **0.6424** | +0.0071 |

İzolasyon hem en düşük hem en az kazanan; %25 ağırlıkla directed seviyesine çıksa toplam MAP +0.017
(LB'de +0.0034). Arkadaşın dokümanlarındaki **tek hiç ölçülmemiş** fikir (E3, `03_ONCELIKLI_DENEY_KARTLARI.md`)
tam olarak buna nişan alıyor.

| 110 | **Kritik geçiş bloğu** (`src/19_isolation_transition.py`, 11 sütun): çift baskı uygulayınca dışarıdaki oyuncuya ne oluyor — kaç kişi hâlâ karar verecekken çekildi, geri alamayacağı kaç bb koymuştu, ve **dışarıdaki çıktıktan sonra çiftin birbirine uyguladığı baskı kesiliyor mu** (karar fırsatı yoksa sıfır değil **null**) | 1.05M satır, 6 sn. Üç tohum kümesi: toplam **0.68111** (referans 0.68119). İzolasyon 0.6424 → ~0.6461 **ama** soft play 0.6713 → 0.6673 | — | — | ELENDİ (toplam nötr) |
| 111 | **Aileye özel özellik kümesi** (`FAM_ONLY=coordinated_isolation:it_`): blok yalnız kendi uzmanına veriliyor, diğerlerini seyreltmiyor | toplam **0.68142** (referans 0.68119), izolasyon 0.6448 | — | — | ELENDİ (fark gürültü) |

### 110-111. Kural 12'nin sınırı
Deney 110 Kural 12'yi bir kez daha doğruladı — blok bir aileyi kaldırıp bir diğerini tam o kadar düşürdü,
yani seyreltme etkisi gerçek. Ama seyreltmeyi kaldırdığımızda (111) geriye kalan kazanç gürültü seviyesinde.
**Mimari kapasiteyi açmak sinyali görünür kılıyor, olmayan sinyali yaratmıyor.** İzolasyonun düşük MAP'i
bir model kısıtı değil: üreteç izolasyon çiftlerine pair başına ~18 el yerleştirip 5'ini listeliyor
(arkadaşın ölçümü), yani seçilemeyen eller gerçekten ayırt edilemez.

| 112 | **İki mimarinin harmanı** (v58 yönlendirilmiş + v55 havuz, çift-içi sıra harmanı; sıra korelasyonu 0.867) | w=0.1/0.2: 0.6811, w=0.3: 0.6809, w=0.5: 0.6729 — v58'in kendisi 0.6811 | — | — | ELENDİ (çeşitlilik var, katkı yok) |

## 18. 20 Eylül günü sonu — nihai durum

**Gönderilmiş:** v55 = **0.91748**. **Hazır aday:** v58, kanıt CV 0.6812 (v55: 0.6631), tahmini **~0.9205**.

Bugün 26 deney yapıldı, **biri** kabul edildi. Kabul edilen tek şey bir özellik değil, bir **mimari kısıt**:
tek havuz lambdarank üç davranış ailesi için tek fonksiyon öğrenmek zorundaydı. Dört ayrı "ELENDİ" kararı
(kronoloji −0.0020, yön tutarlılığı −0.0049, çift-içi normalizasyon −0.0020, aile parametreleri nötr) o kısıt
kalkınca **+0.0137**'ye, kesme seviyesiyle birlikte **+0.0181**'e döndü.

**Ölçülerek kapatılan eksenler:** sinir ağı/transformer/latent (arkadaşın dört denemesi), kolüzyon halkaları
(5.6σ gerçek, kullanılabilir kazanç yok), faz karşıtlığı (AUC 0.48 ile gerçek, kazanç yok), ilk beşi yeniden
sıralama (24 kural düz), çift modelinde uzman mimarisi (yönü belirsiz), dördüncü aile risk tabanı (dürüst
protokolde her f'de zarar), izolasyon kritik-geçiş bloğu (nötr), iki mimari harmanı (katkı yok),
aileye özel kapasite ve kesme (gürültü), 100 aday, yumuşatılmış uzmanlar.

### Günün üç kuralı
> **Kural 12.** Bir bloğun tek başına ölçülen değeri, modelin o bloğu kullanabilecek kapasitesi yoksa
> anlamsızdır. Blok elemeden önce mimarinin o bloğu ifade edebildiğinden emin ol.
>
> **Kural 13.** Bir taramadan çıkan en iyi okuma, taramada kullanılmayan bir tohum kümesinde tekrarlanmadıkça
> gerçek değildir. (+0.0058 okunan ayar bağımsız kümede +0.0003 çıktı.)
>
> **Kural 14.** İki ayrı popülasyona karşı hesaplanmış AP'yi ağırlıklı toplayarak "beklenen metrik" üretme;
> metrik tek sıralama üzerinden tanımlıysa simülasyon da tek sıralama üzerinde olmalı. (İşaret tersine döndü.)

### Gönderim planı (4 ortak hak, bitiş 20 Eylül 22:00 UTC)
1. `submission_v58.csv` — tek ve izole değişiklik (risk/davranış v55 ile bit düzeyinde aynı), tahmini +0.0036.
2. v58 doğrularsa `submission_v58_amb.csv` — dördüncü aile kuralını (487 çift) yeni kalibrasyonda sınar.
3. Kalan iki hak arkadaşa; kota ortak.
4. **Final iki dosya elle seçilmeli** (MCP'de bu işlem yok): en yüksek iki public skor.

Üç dosya da resmî puanlayıcı uç durum denetiminden geçti: tekrarlı kanıt eli 0 (resmî tarafta gönderim
hatası sayılıyor), `NO_EVIDENCE` 0, üç aile de dolu, risk sonlu, 112.540 tekil çift.

## 19. Son gönderimlerin okunması — takım rekoru artık 0.91834

Bugünün üç gönderimi tek değişkenli temiz bir deney oluşturuyor: üçünde de risk ve davranış sütunları
bizim v55'imiz, **birebir aynı**; değişen yalnız kanıt sütunu.

| Saat (UTC) | Kim | Kanıt sütunu | LB |
|---|---|---|---|
| 01:55 | B | B'nin kanıt motoru v1 | 0.91233 |
| 00:01 | A | **bizim v55 kanıtımız** | **0.91748** |
| 12:58 | B | B'nin kanıt motoru v2 (noisy-OR/LSE aksiyon havuzlaması) | **0.91834** |

Kanıt ağırlığı 0.20 olduğu için: B'nin en iyi kanıt motoru bizim v55'imizden yalnız **+0.004 MAP** önde.
Bizim v58'imiz v55'ten **+0.018 MAP** önde (üç tohum kümesinde doğrulandı) → v58 ≈ **0.9211**,
mevcut takım rekorunun **+0.0028** üstü. Bu artık formül tahmini değil, LB'de ölçülmüş bir zincire dayanıyor.

| 113 | **Noisy-OR ve log-sum-exp aksiyon havuzlaması** (`src/04c_action_model.py`, 5 sütun: `act_nor`, `act_nor_l`, `act_lse10/25/50`) — B'nin v1→v2 sıçramasının (+0.006 LB) açıklamasında yazan yöntem | aynı yeniden eğitilmiş model üstünde A/B, üç tohum kümesi: havuzlama yok **0.68119**, var **0.67944** (0.6792 / 0.6828 / 0.6763) | — | — | **ELENDİ (−0.0018, üstelik oynak)** |

### 113. Neden onlarda +0.006, bizde −0.002
Bizim aksiyon modelimiz el seviyesine zaten yedi ayrı yoldan çıkıyor: `act_max`, `act_2nd`, `act_n05`,
`act_mean`, `act_sum`, `act_max_pre/post`, `act_max_bactive`. Noisy-OR bunların üzerine yeni bir okuma
getirmiyor; `act_lse*` ise düşük skorlarda pratikte `τ·log(n)`'e yakınsıyor, yani aksiyon sayısının
gürültülü bir tekrarı oluyor. Deney 74'ün (bahis boyutu) ve deney 80'in (kronoloji) dersinin üçüncü kez
tekrarı: **başka bir hattın kazancı, kendi hattında aynı boşluk yoksa transfer etmiyor.**

**Yan bulgu:** A/B'nin kontrol kolu eski referansı 12 haneye kadar birebir üretti
(0.6807803166069296 vs 0.6807803166069294), yani `04c_action_model.py` tam belirleyici ve v58
yeniden üretilebilir. (Belirleyici olmayan tek adım `05_pair_model.py`.)

## 20. Graf teorisi ekseni — üçüncü ve son deneme, kapandı

Bugün grafik ekseni üç ayrı nesne üzerinde denendi (bizde ikisi, arkadaşın deposunda iki tanesi daha):

| # | Nesne | Sonuç |
|---|---|---|
| 79 | **Oyuncu grafiği** (`src/16_graph_pair.py`): hub, dışlayıcılık, üçgen desteği | etiketli AP 0.99856 → 0.99842; evren medyanı 278 → 271 — ELENDİ |
| 114 | **Kurban grafiği** (`src/20_victim_graph.py`): çift–dışarıdaki oyuncu ikili grafiği, 11 sütun | aşağıda — ELENDİ |
| (B) | `80_player_graph_features.py` (OOF risk ağırlıklı oyuncu grafiği) | temiz AP 0.9693 → 0.9693 — ELENDİ |
| (B) | `300_relational_graph_residual.py` (400 kapalı havuzda iki yönlü medyan-cila kalıntısı, yıldız cezası, üçgen ve ortak-komşu desteği) | "hiçbir sütun üç bant genişliğinde de null'ını geçmiyor... görünür kazancın tamamı sızıntı, dosya yazılmadı" |

### 114. Kurban grafiği (`src/20_victim_graph.py`)
Kenar ağırlığı: çiftin birlikte kârda olduğu ellerde dışarıdaki oyuncunun kaybettiği bb. 9.77M el,
5.33M çift–kurban kenarı, 7 saniye. İki soru: **yoğunlaşma** (hep aynı kurban mı) ve **paylaşım**
(o kurban, çift üyelerinin *diğer* ortakları tarafından da soyuluyor mu — veri kümesinde üç oyuncu
gerektiren tek ifade).

| Sütun | AUC (1860 etiketli çift) |
|---|---|
| `vg_drained_bb` (toplam çekilen) | 0.6220 |
| `vg_n_victims` | 0.5663 |
| `vg_hhi` (yoğunlaşma) | **0.4249** — ters yönde |
| `vg_top_share` | **0.4229** — ters yönde |
| `vg_shared_p` / `vg_shared_q` (paylaşım) | 0.5209 / 0.5627 |
| `vg_nshare_max` | 0.5284 |

Çift modelinde (3 tohum): etiketli AP 0.99862 → **0.99848**, ilk-200 139.3 → 137.3, medyan ~275 (değişmiyor).

**İki şey öğretici.** Birincisi, tek anlamlı sütun (`vg_drained_bb`) grafik değil: "bu çift çok para
kazanıyor" diyor ve bunu model zaten biliyor. İkincisi, yoğunlaşma **ters** çıktı — kolüzyon çiftleri
kazançlarını daha *çok* kurbana yayıyor, tek bir marka dönmüyorlar. Hipotezin tersi.
Gerçekten üç-oyunculu olan tek ifade (paylaşım) AUC 0.50-0.56, yani şans.

### Graf teorisi hakkında nihai karar
Yapı **gerçek ama bilgi taşımıyor**: 49 oyuncu birden fazla pozitif çiftte (null 25.4 ± 4.2, **5.6σ**),
**kapalı üçgen sıfır**, ve bilinen 372 pozitifin hepsi taban özelliklerle zaten ilk 1000'de.
Sebep basit: X hem Y hem Z ile kolüzyon yapıyorsa iki çift de davranışsal imzayı doğrudan gösteriyor;
grafik ancak doğrudan kanıtın zayıf olduğu yerde öncelik bilgisi katardı ve oradaki 2× oran AP'yi
kıpırdatmıyor. **Üreteç kolüzyonu ikili yerleştiriyor; keşfedilecek bir grup yapısı yok.**

## 21. Kapanış: v58 geç gönderim = 0.92121

| 115 | **v58 (geç gönderim)**: yönlendirilmiş aile uzmanları (havuz 0.10 + üç uzman 0.90, davranış kafasının kalibre olasılıklarıyla yönlendirilmiş), kesme 50, 12 tohum, artı temas kronolojisi + çift-içi normalizasyon + bırak-birini-dışarıda yön tutarlılığı. Risk ve davranış sütunları v55 ile bit düzeyinde aynı | kanıt CV 0.6631 → **0.6812** (üç bağımsız 6-tohumluk kümede) | **0.92121** | **+0.00373** | ALINDI |

### Tahmin, ilk kez birebir tuttu
`0.91748 + 0.20 × 0.0181 = 0.92110`, gerçek **0.92121**, artık **+0.00011**.
Önceki en iyi artık ±0.0018 bandındaydı; v55'te ise +0.0099 ile bandı beşe katlamıştı.
Fark, tasarımın kendisi: v58 yalnız kanıt sütununu değiştirdi ve `0.7P + 0.1B` terimini bit düzeyinde
sabit tuttu, yani formülün sabiti (0.7843) geçerli kaldı. **Tahmin formülü, değiştirdiğin terimi izole
ettiğinde tam çalışıyor; iki terimi birden değiştirdiğinde çalışmıyor.** v55'in büyük artığı bir gizem
değil, o kuralın ihlaliydi.

### Sıralama maliyeti
Yarışma bitiş tablosunda takım (Brothers) **0.91834 ile 15.**. v58'in 0.92121'i zamanında gönderilse
**10.** olurduk (10. 0.92090, 11. 0.92060). **Beş sıra, gönderilmeyen bir dosya yüzünden kayboldu.**
Dosya bitişten ~13 saat önce hazır ve doğrulanmıştı; gönderim onayı beklendi.

| | public |
|---|---|
| 1. | 0.94068 |
| 3. (ilk üç sınırı) | 0.93894 |
| 10. | 0.92090 |
| **v58 (geç)** | **0.92121** |
| 15. Brothers (resmî) | 0.91834 |
| v55 (bizim en iyi zamanında gönderimimiz) | 0.91748 |

### Günün bilançosu
~35 deney, **bir** kabul. Kabul edilen şey bir özellik değil, bir mimari kısıtın kaldırılması: tek havuz
lambdarank üç davranış ailesi için tek fonksiyon öğrenmek zorundaydı. Ayrı ayrı "ELENDİ" damgası yemiş
dört blok (kronoloji −0.0020, yön tutarlılığı −0.0049, çift-içi normalizasyon −0.0020, aile parametreleri
nötr) o kısıt kalkınca +0.0137, kesme seviyesiyle birlikte **+0.0181** verdi — ve LB'de +0.00373 olarak
birebir göründü.

Ölçülerek kapatılan eksenler: graf teorisi (üç nesne, dördü de sıfır), sinir ağı/transformer/latent,
faz karşıtlığı, ilk beşi yeniden sıralama, çift modelinde uzman mimarisi, dördüncü aile risk tabanı,
izolasyon kritik-geçiş bloğu, noisy-OR/LSE havuzlama, iki mimari harmanı, aileye özel kapasite ve kesme,
kurban grafiği, 100 aday, yumuşatılmış uzmanlar.

## 22. Devir planının (`docs/research_20260918/04_AGENT_DEVIR_PLANI.md`) uygulanması

Doküman bir **süreç planı**: doğrulama paketleri (A–E), takvim, deney şablonu, final üretim kabul ölçütleri.
İçinde doğrudan uygulanacak bir model tarifi yok; uygulanabilir iki maddesi var.

### Paket A — doğrulama (dokümanın "ilk iş"i)
§7'nin 3. maddesi kritik bir uyarı: onlarda aynı tohumla farklı masa listelerinden fold üretmek tutarsız
(245 pozitif masanın 199'unda fold değişiyor) ve düzeltme kanıt OOF'unu **0.6930 → 0.6817** düşürmüş,
yani ölçümleri +0.011 şişkinmiş.

**Bizde bu sorun yapısal olarak yok.** Beş aşamanın hepsi `tidx % NF` kullanıyor — masa indeksinin
deterministik fonksiyonu, aralarında hiçbir fark yok: `04_hand_scorer.py:41`, `04b_stage1b.py:56`,
`04c_action_model.py:105`, `05_pair_model.py:127`, `05b_evidence_reranker.py:154`. Bir çiftin bütün elleri
tek masada olduğu için hiçbir çift fold sınırını aşmıyor. Aşama-1 skorlarının hepsi fold dışı üretiliyor
(04 docstring'i, 04b:114-124, 04c:191). **Bugünün rakamları bu açıdan sağlam.**

### Deney kartı E6 — eşleştirilmiş sahte-partner null kalibrasyonu
Kartın hiç ölçülmemiş fikri. `src/21_matched_null.py`: her yönlü çift (A→B) için A'nın *diğer* rakipleri
üzerinden bir null kuruluyor, ama **maruziyet eşleştirmeli** — yalnız paylaşılan karar sayısı 1.5 kat
bandında olan rakipler havuza giriyor, çünkü istatistiğin yayılımı maruziyetle daralıyor ve 300 elde
görülen rakiple 30 elde görülen rakip karşılaştırılabilir okuma değil. Çıktı: ampirik p-değeri, havuz
merkezinden MAD-ölçekli uzaklık, en iyi rakibe göre fark. 7.07M eşleştirilmiş karşılaştırma, boş havuz %2.4.
(`src/16_graph_pair.py`'den farkı: orada eşleştirme yoktu.)

| 116 | **E6 eşleştirilmiş null** (`src/21_matched_null.py`, 24 sütun) | tek başına en iyi `mn_lead_oa_surp_excess_hi` AUC 0.9874 — ama taban `oa_g_tail_sum` 0.9932'nin **altında**. Çift modelinde (3 tohum): etiketli AP 0.99862 → **0.99826**, ilk-200 139.3 → **134.0** | — | — | **ELENDİ** (kartın kendi kill kuralı: "mevcut LLR'ı aynen yeniden üretiyorsa bırak") |

### İkinci geç gönderim: v58_amb
§4 ikinci seçim için "ancak ölçülmüş dayanıklılık/çeşitlilik gerekçesi varsa farklı recipe" diyor.
Elimizdeki tek gerçekten farklı tarif davranış sütununu değiştiren `submission_v58_amb.csv`.

| 117 | **v58_amb (geç gönderim)**: v58 + dördüncü aile kuralı — risk sıralamasının ilk 1500'ü içinde davranış kafasının en yüksek sınıf olasılığı 0.60'ın altında olan **487 çift** `other_coordination` olarak yeniden etiketlendi. Risk ve beş kanıt sütunu v58 ile bit düzeyinde aynı, yani yalnız davranış sütunu izole ediliyor | — | **0.92197** | **+0.00076** (v58 üstüne) | **ALINDI — nihai en iyi** |

### 117. Kural v55 kalibrasyonunda da, v58'de de kazandırıyor
v55 kalibrasyonunda aynı kural 326 çiftle +0.0013 getirmişti (deney 60); v58'de 487 çiftle +0.00076.
Davranış ağırlığı 0.10 olduğu için bu BehaviorMAP'te +0.0076 demek. Kural, kanıt tarafı 0.0181 iyileştikten
sonra da ayakta — yani dördüncü aile sinyali kanıt kalitesinden bağımsız.

## 23. Nihai tablo

| Sürüm | Ne değişti | LB |
|---|---|---|
| v45 | rakip-farkında blok yok | 0.90858 |
| v55 | rakip-farkında iki blok (zamanında gönderilen en iyimiz) | 0.91748 |
| v58 | yönlendirilmiş aile uzmanları + üç blok + kesme 50 (geç) | 0.92121 |
| **v58_amb** | **+ dördüncü aile kuralı (geç)** | **0.92197** |

Bugünün toplam kazancı **+0.00449** (v55 → v58_amb). Resmî bitiş tablosunda takım 0.91834 ile **15.**;
0.92197 zamanında gönderilse **10.** olurduk (10. sıra 0.92090, 9. sıra 0.92501).
İlk üç sınırı 0.93894 idi ve bugünün bulgularıyla erişilebilir değildi.
