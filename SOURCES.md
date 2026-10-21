# PermitForge sources and licences

This is a provenance ledger for the preserved inputs, not a request to fetch a new
snapshot. URLs identify upstream locations; SHA-256 identifies the actual input.
The v2 conversion reuses the v1 scenarios and meteorology. The extraction is under
`convert/v1-src/` and is deliberately excluded from git; task manifests record file
hashes. See [convert/README.md](convert/README.md) for preservation and integrity checks.

Source abbreviations used below (all were read without modification):

- **G**: `results/evidence/provenance/scenario-generator.py.txt`.
- **C**: `results/evidence/provenance/generator-container.txt`.
- **E**: `results/evidence/provenance/runtime-source.yaml`.
- **X**: `convert/v1-src/EXTRACT_MANIFEST.json` (source git identity, file-to-destination mapping and SHA-256).

## Rights and attribution

The NOAA observations and EPA model inputs are identified as U.S. Government works
in the source declarations (E and `results/evidence/provenance/scenario-source.yaml`).
Under [17 U.S.C. §105(a)](https://www.law.cornell.edu/uscode/text/17/105), U.S. Government
works do not receive U.S. copyright protection. This is the public-domain basis for
these inputs; the code licence does not relicense upstream government data.
Agency terms: [NOAA/NWS disclaimer](https://www.weather.gov/disclaimer) and
[EPA disclaimers](https://www.epa.gov/web-policies-and-procedures/epa-disclaimers).
No agency endorsement is implied.

PermitForge code is licensed under [Apache-2.0](LICENSE), Copyright 2026 Attoloop.
Third-party Python dependencies retain their licences; package pins are in the
root, core and OpenEnv `uv.lock` files. Preserved generator/source snapshots and
original relative paths with hashes are in
[provenance/origins.json](results/evidence/provenance/origins.json).

## NOAA observations and processed meteorology

The source is NOAA NCEI ISD hourly observations for 2020-2024 and IGRA2 soundings,
selected by station. G:20 defines station metadata, G:21 is the `ASSETS` digest table,
G:58-61 constructs the download URLs, and G:65-84 verifies downloads against those
pins. G:150-164 concatenates ISD years and filters IGRA2 records to the same years.
IGRA2 ZIP hashes cover the whole downloaded archive, including records outside the
selected years. These are recorded acquisition pins; the raw NOAA archives have not been re-downloaded or re-hashed for this release.

Reproduce the station rows and every raw URL/digest below, without importing or
running the generator (run from the repository root):

```bash
uv run python - <<'PYNOAA'
import ast
from pathlib import Path
p = Path('results/evidence/provenance/scenario-generator.py.txt')
values = {}
for node in ast.parse(p.read_text()).body:
    if isinstance(node, ast.Assign):
        for name in node.targets:
            if isinstance(name, ast.Name) and name.id in ('STATIONS', 'ASSETS'):
                values[name.id] = ast.literal_eval(node.value)
for s in sorted(values['STATIONS'], key=lambda s: s['id']):
    print(s)
    names = [f"{s['usaf']}-{s['wban']}-{y}.gz" for y in range(2020, 2025)]
    names.append(s['ua'] + '-data.txt.zip')
    for name in names:
        url = (f'https://www.ncei.noaa.gov/pub/data/noaa/{name[-7:-3]}/{name}'
               if name.endswith('.gz') else
               'https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/' + name)
        print(url, values['ASSETS'][name])
PYNOAA
```

### Station identities

USAF and WBAN are the surface-station identifiers; IGRA2 identifies the upper-air
station, which can be at a different location. `tz` is the positive westward offset
passed to AERMET, not a daylight-saving timezone. Metadata: G:20.

| Station | USAF | WBAN | IGRA2 | Land-use assumption | Wind height (m) | tz |
|---|---|---|---|---|---:|---:|
| KABQ | 723650 | 23050 | USM00072365 | arid_grassland | 10.05 | 7 |
| KALB | 725180 | 14735 | USM00072518 | grassland | 10.05 | 5 |
| KAMA | 723630 | 23047 | USM00072363 | arid_grassland | 10.05 | 6 |
| KCRP | 722510 | 12924 | USM00072251 | grassland | 10.05 | 6 |
| KDDC | 724510 | 13985 | USM00072451 | arid_grassland | 7.92 | 6 |
| KELP | 722700 | 23044 | USM00072364 | arid_grassland | 10.05 | 7 |
| KGRB | 726450 | 14898 | USM00072645 | grassland | 10.05 | 6 |
| KIAD | 724030 | 93738 | USM00072403 | grassland | 10.05 | 5 |
| KJAX | 722060 | 13889 | USM00072206 | grassland | 10.05 | 5 |
| KMSP | 726580 | 14922 | USM00072649 | grassland | 10.05 | 6 |
| KOMA | 725500 | 14942 | USM00072558 | grassland | 10.05 | 6 |
| KPIT | 725200 | 94823 | USM00072520 | grassland | 10.05 | 5 |
| KSHV | 722480 | 13957 | USM00072248 | grassland | 10.05 | 6 |
| KTOP | 724560 | 13996 | USM00072456 | grassland | 7.92 | 6 |
| KTUS | 722740 | 23160 | USM00072274 | arid_grassland | 10.05 | 7 |

### Original download pins

Every row is a NOAA U.S. Government input under the rights statement above.

| Station | File URL | SHA-256 of downloaded bytes |
|---|---|---|
| KABQ | [723650-23050-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/723650-23050-2020.gz) | `fd34fbcfb8dcd6ab4aa4f3fad77e6f10db3096d23f8ecf73e08568ca5be9bd62` |
| KABQ | [723650-23050-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/723650-23050-2021.gz) | `ffbee1af6e641222f8908892e81698c1b83cf76220a9757320863577804e71b7` |
| KABQ | [723650-23050-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/723650-23050-2022.gz) | `088c1d3e546532a7010e22b4edc81df76b9c2e9759b96f7a4ca860e8176d8db7` |
| KABQ | [723650-23050-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/723650-23050-2023.gz) | `f730a9a549706d97ea4da8a98e369a8c843eff8622e8083026f16282e2ce2117` |
| KABQ | [723650-23050-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/723650-23050-2024.gz) | `d55bcd9b6e4ac9c126543a1048dc8a316c95cf4209510649c071d9c7da4c558e` |
| KABQ | [USM00072365-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072365-data.txt.zip) | `c366efacf62e2988c8b17a5e2c0803cbfc206b4215738d9005424a161288b289` |
| KALB | [725180-14735-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/725180-14735-2020.gz) | `852a530e37e4bdb15c70da7963567c7e65a39c5da884dde91cc728376a25ddcb` |
| KALB | [725180-14735-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/725180-14735-2021.gz) | `377e1f1984d832f00a967a6ae4e6054f9dbda1688a8b957384ffa7af7a9c610e` |
| KALB | [725180-14735-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/725180-14735-2022.gz) | `db60296ff803e307f1f3989f1812ffcfe1ee044974d6c377ec7f25a43feea401` |
| KALB | [725180-14735-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/725180-14735-2023.gz) | `3ef72c2fce410daae49b71e550086ab0ccd2d2fac947ae7e47c60b8c5ea3c9f5` |
| KALB | [725180-14735-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/725180-14735-2024.gz) | `2f31e6ee6353736d42721be1e273448af053aa1196e83645a8a0d17ee13f7dfd` |
| KALB | [USM00072518-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072518-data.txt.zip) | `21dcbc93dd30c837886295db28cd35620c6e19899055cb7a2664100797b1d305` |
| KAMA | [723630-23047-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/723630-23047-2020.gz) | `4c28415f5610f46923a31816a8fd15376c873baac4714ec2b85f6551d2e2053c` |
| KAMA | [723630-23047-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/723630-23047-2021.gz) | `1290553beaabf6ff684b84ba6275414fe3dbe098ca86989d62d48a64b40b7099` |
| KAMA | [723630-23047-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/723630-23047-2022.gz) | `cf877baf41021e0970cc5278c67326c956a2b554ee9b24462c0b4833e1f4565b` |
| KAMA | [723630-23047-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/723630-23047-2023.gz) | `bd988970c297b2fc1cd51ee872307773a387a271002e1b108cf0895f74b74cfd` |
| KAMA | [723630-23047-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/723630-23047-2024.gz) | `99cd29d4b46d75c58bb442185d4374977dbb8952ad6a799e1e640055145dcd79` |
| KAMA | [USM00072363-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072363-data.txt.zip) | `d7a5ddbce3491eec03f5045b1e71f4662b91db6c892d23a75ef656e401c3e74f` |
| KCRP | [722510-12924-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/722510-12924-2020.gz) | `c51361ebe0e2b61245301eddfedfdb06c271c2a0dd6445ef89e5bcd338872c3b` |
| KCRP | [722510-12924-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/722510-12924-2021.gz) | `dabc70f66f7adb9552da30afe973106f3f6e46bf4c69021f3ffed7f4a6277543` |
| KCRP | [722510-12924-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/722510-12924-2022.gz) | `23e46167620b04f0a448ff15677fc9da4d21f1c9b800656eb4ae0d0b8d0e7877` |
| KCRP | [722510-12924-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/722510-12924-2023.gz) | `e78c75094f17ad106bb695a7f2e83ac482a21b12ffe3e2eebb724485c67583d7` |
| KCRP | [722510-12924-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/722510-12924-2024.gz) | `d416111e9ff9b54d5afa735d469e5d469aa2e8803c863ccd131d43c04519c4e1` |
| KCRP | [USM00072251-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072251-data.txt.zip) | `7bd45a043b190ed71f4df46965160c71c60f57f5a49ce1d2731a515268f8ddc9` |
| KDDC | [724510-13985-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/724510-13985-2020.gz) | `d4fade01904043d552638fe1e3082617441929eea643c2fea42b2955960f9a7c` |
| KDDC | [724510-13985-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/724510-13985-2021.gz) | `443dff72bee29beef89d5fb00e53c06830b192c29a0dc3fc7d2806c8ee65e48e` |
| KDDC | [724510-13985-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/724510-13985-2022.gz) | `d187e718683b0dfcc264cc67bdb892040b50b108e8861878545024fb32ccc099` |
| KDDC | [724510-13985-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/724510-13985-2023.gz) | `2f43fe0d70d85c739f57ae46079dd67f90141862ec85cacc4f913e3b02d3e4d0` |
| KDDC | [724510-13985-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/724510-13985-2024.gz) | `827f5733c3ef14c6572ef08fcd10e7fcc721795e99ad098ff051850337fdfa35` |
| KDDC | [USM00072451-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072451-data.txt.zip) | `2f56eab4a247e7205c761d0a21bb4b9957816643c090f06b222373f4f262c2c6` |
| KELP | [722700-23044-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/722700-23044-2020.gz) | `81e78237232ccfbc2bd810609f12d082f7492a9323301f1914fd58d62b139ae3` |
| KELP | [722700-23044-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/722700-23044-2021.gz) | `06b004920a045a2c04f61f7b403a8a78feeaa60ad970790b9383c3cb65fe1165` |
| KELP | [722700-23044-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/722700-23044-2022.gz) | `114215e2e02c0aa75223143607b7dc8a55a84727c31f8589f7979daf33bf7a41` |
| KELP | [722700-23044-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/722700-23044-2023.gz) | `cbb0f701776503be4456d05a723757002f9a52e5ffe24517b7a2007349fd71cb` |
| KELP | [722700-23044-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/722700-23044-2024.gz) | `95b573c8f014c683c6842c03c6b8e03b6241a3ffa30f881fdd6f7d36a1b7ce73` |
| KELP | [USM00072364-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072364-data.txt.zip) | `1a58f948d46c380f6ff8c711e1b51c42d39b63b63205291acf7561c6e265450c` |
| KGRB | [726450-14898-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/726450-14898-2020.gz) | `f9a3bf80a13cf39afc14db1f100701951e9fd54ba001d67ac57fce129e655696` |
| KGRB | [726450-14898-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/726450-14898-2021.gz) | `336ccb4918cf202a1f792c5762949a1b392c741ba1c3c88a3ce61b90d4bd5fbd` |
| KGRB | [726450-14898-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/726450-14898-2022.gz) | `0af444b44a30168a2b2e1da2b366e5b470c5130fe8c1091c7d882d492fa79147` |
| KGRB | [726450-14898-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/726450-14898-2023.gz) | `15c1dd554aef60b99060af4976383f9ed84ed983cbaf2d812a23dc623ed54362` |
| KGRB | [726450-14898-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/726450-14898-2024.gz) | `2ef3497f02152b54843c29b508feb95758428f3facf84d2730913418c2d162bf` |
| KGRB | [USM00072645-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072645-data.txt.zip) | `f8402ffad5e402c985fd43ca2748865d2d1b8431c803d52f31b6073662dcc8ef` |
| KIAD | [724030-93738-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/724030-93738-2020.gz) | `24ff1774dbb25ce9c8948e28c3790c191d03e2d8582ed900d2e028090e2f7568` |
| KIAD | [724030-93738-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/724030-93738-2021.gz) | `4bee4c4802a332861e8b3578bf4aeb79442d358c36592197e2f201e29f66f60a` |
| KIAD | [724030-93738-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/724030-93738-2022.gz) | `35e5b7ed6861583fa66eddef46ad904b4e8cf1fda95b0d90f25ddcb91b553c3e` |
| KIAD | [724030-93738-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/724030-93738-2023.gz) | `31e8e73f3431daa815de9307e7a37ab1c2d5a8bf26668db1e5291f4f5ef9b066` |
| KIAD | [724030-93738-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/724030-93738-2024.gz) | `e409da3b649354f7140a7003a7db167a49ffe58566a67d2defd016e804b3ad52` |
| KIAD | [USM00072403-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072403-data.txt.zip) | `a86374f2749b2b0701246dca48f7924fd6789357fbe188ea8868af0ec72d3f70` |
| KJAX | [722060-13889-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/722060-13889-2020.gz) | `9f19cbc65011d29fc9d9c207ee0648d8f58ce301b5a4c62625ba4bf78e9e77e8` |
| KJAX | [722060-13889-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/722060-13889-2021.gz) | `f854bd3aaf07177347cd8c8c923685d5cc2001e0055296df960632cca76702dd` |
| KJAX | [722060-13889-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/722060-13889-2022.gz) | `704cbfc0463daea169d25646505ff98ad4b850f8ab89d668350bd831a9be5cff` |
| KJAX | [722060-13889-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/722060-13889-2023.gz) | `f2b6447975dee67a5304c651940ad85e87b18a17fb3dcb2a766f160ce25e501b` |
| KJAX | [722060-13889-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/722060-13889-2024.gz) | `0c7b713ea75254d577f8301d2d98792774b339f2d0bada69ddce80482e04ee6d` |
| KJAX | [USM00072206-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072206-data.txt.zip) | `b15749289a496ba1a2148d551e68dc6e5ee9e9e3352034235a1ffb50f66e1515` |
| KMSP | [726580-14922-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/726580-14922-2020.gz) | `a8f40a756b3afd93c66bc18ee809538e95578170140be800c885a3c30fc3d3d5` |
| KMSP | [726580-14922-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/726580-14922-2021.gz) | `7fbd5800d87b31947497719d2d6d65fc0719dfff5d1b1747fd7a566cb7a77f3e` |
| KMSP | [726580-14922-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/726580-14922-2022.gz) | `cd323eed354a80118c691dfc5896335e117912c3f8aaba87481336245325f093` |
| KMSP | [726580-14922-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/726580-14922-2023.gz) | `f712fc46365fc1d5759ea94cd294e9f69e1c538ee9d884fd8e62ba35d3b0e15e` |
| KMSP | [726580-14922-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/726580-14922-2024.gz) | `53b8d3ead33d31cfc823a544437de1ffbee59e50b8d455a193c4a43d1bec055d` |
| KMSP | [USM00072649-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072649-data.txt.zip) | `2f506d2fcc1279c348fa1550c943728b1f0498b4d969fc98a7be64a3dff58ae0` |
| KOMA | [725500-14942-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/725500-14942-2020.gz) | `4eca08a43129ad53415f45f3937247daa3d20f8a25cbf433d943f6962203d2eb` |
| KOMA | [725500-14942-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/725500-14942-2021.gz) | `428cad1f9535cb85d7a98148a60872524162c2d1a0f2dde284661ed08c38bed5` |
| KOMA | [725500-14942-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/725500-14942-2022.gz) | `8af4c438f4ef63a14e874e65f9e1b202ff135353d618cb5f3672f2d1883b0e52` |
| KOMA | [725500-14942-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/725500-14942-2023.gz) | `df9168f357baf8bd047dde6ad0c305ac277ee566cb4993b773af4f2a9aa890cb` |
| KOMA | [725500-14942-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/725500-14942-2024.gz) | `99925770a1a7747beb37f341f6c4830870fc0f385825be66bf41f64cfe2f9938` |
| KOMA | [USM00072558-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072558-data.txt.zip) | `f009b9c8eef44e348519921887d8bbe322b38b389143ef040209fdc1ee510e49` |
| KPIT | [725200-94823-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/725200-94823-2020.gz) | `de580efda4ea5137846e041db41dfa79343e3e93991533a06068c7cb918c50ae` |
| KPIT | [725200-94823-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/725200-94823-2021.gz) | `9b43ae524593f43105ecc5ac36aca8a7cd1659af3265df01023e575995171d5a` |
| KPIT | [725200-94823-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/725200-94823-2022.gz) | `1808022647e19180605541d7f211dc2c139ae5054b233e87da96415eac66f060` |
| KPIT | [725200-94823-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/725200-94823-2023.gz) | `31aa8148f89e4cf76852fce69e8baffc0d0b6e8ce23a6bedffeadf0b7bae1c1c` |
| KPIT | [725200-94823-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/725200-94823-2024.gz) | `8ad6fcdbaf85e24660fc53238ef64199f6da43847e74f1f60f11d9822061b43f` |
| KPIT | [USM00072520-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072520-data.txt.zip) | `a0ba44f84b8929bbc6edb916e332bbca24a8a6aeb4e94035eb83a20991bb4ff6` |
| KSHV | [722480-13957-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/722480-13957-2020.gz) | `e5ebb657220602a4b9c01f73f0108a1c004cf97914cc8273063a9f7f5ac6f6a0` |
| KSHV | [722480-13957-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/722480-13957-2021.gz) | `ad6af23dd667f388d9dfcf38aa1f89518e150ea812a934632e029f716e22bccf` |
| KSHV | [722480-13957-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/722480-13957-2022.gz) | `379a7155ad71526af6ad088813900d31121c40076b2c4816e2d26a98da093dc3` |
| KSHV | [722480-13957-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/722480-13957-2023.gz) | `2b29d01d0611d9bdd6600a5b21525679045d29ae056971c3106700d015b80943` |
| KSHV | [722480-13957-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/722480-13957-2024.gz) | `f0035ebacc46b3a6790b8ea0517a5922b7e108ac13b8a22313ef9ee82beb95c6` |
| KSHV | [USM00072248-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072248-data.txt.zip) | `cff6c5f44f23699d01489f2d5b80b4b122b74a5a2bd4cd7f82b222f0fba7a20d` |
| KTOP | [724560-13996-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/724560-13996-2020.gz) | `f78f63eac39ea3ba743cca4f29dfef0dbd9290de6a3269cb85b34b8677322415` |
| KTOP | [724560-13996-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/724560-13996-2021.gz) | `feecf019893f379acf7183862ba78c317132a7ce270e401a82b40cfc1e7817f4` |
| KTOP | [724560-13996-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/724560-13996-2022.gz) | `ac5c861c9753b522006cd5b7de3c156ff65686f66b991925c98e5b09a546e3d7` |
| KTOP | [724560-13996-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/724560-13996-2023.gz) | `38f06f38639d0ee8c3625eabf7f9f1cf8dcfb0845f0d8afe1bfcdac39888d068` |
| KTOP | [724560-13996-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/724560-13996-2024.gz) | `f8ea66e126bd8c975c50a984eeb62badd42b5562a24e5152e0383936f643397b` |
| KTOP | [USM00072456-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072456-data.txt.zip) | `d25092b12d73cd3f2a52fa7ff8a320178811754aaac7bbbe03e718ba2b44f229` |
| KTUS | [722740-23160-2020.gz](https://www.ncei.noaa.gov/pub/data/noaa/2020/722740-23160-2020.gz) | `e10a10c4c0ffd938cfdf64d0ac1aefdd62c1df46157fa34f2a313a77fbd80c20` |
| KTUS | [722740-23160-2021.gz](https://www.ncei.noaa.gov/pub/data/noaa/2021/722740-23160-2021.gz) | `eec51b652679e2b524623e74f14eb57c8ca7ed5e48665a0750ba290d0f02c13e` |
| KTUS | [722740-23160-2022.gz](https://www.ncei.noaa.gov/pub/data/noaa/2022/722740-23160-2022.gz) | `50b8cc294c4650c79ae859efd76b361c4525d3aed45220c35e1af2c1b26ce1ed` |
| KTUS | [722740-23160-2023.gz](https://www.ncei.noaa.gov/pub/data/noaa/2023/722740-23160-2023.gz) | `bc00cab00e4fd8293dabf1e609a563484f21e787c0393e8c5bf8191fd5677c8b` |
| KTUS | [722740-23160-2024.gz](https://www.ncei.noaa.gov/pub/data/noaa/2024/722740-23160-2024.gz) | `d92ca248111ab9b3542065fe7531a5fb4db9ad393e404bf616840c49cbb3470c` |
| KTUS | [USM00072274-data.txt.zip](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/USM00072274-data.txt.zip) | `a5bdd85d486c90c0cf21e9de7a45d2ac6adca46643a4ec24bda3cbf950ceec11` |

### AERMET processing assumptions

AERMET 26135 reads surface `ISHD` and upper-air `IGRA`; the upper-air path includes
`MODIFY`. Both inputs and `METPREP` use `XDATES 2020/1/1 TO 2024/12/31`.
The `.sfc` / `.pfl` files are model-ready derived weather, not raw observations.
The generator validates the surface record count and endpoint years (G:189-195).

The settings are `METHOD WIND_DIR RANDOM`, `METHOD REFLEVEL SUBNWS`,
`FREQ_SECT ANNUAL 1`, `SECTOR 1 0 360`, and one fixed `SITE_CHAR` triplet.
Grassland uses albedo 0.20, Bowen ratio 0.75, roughness 0.10 m;
arid grassland uses 0.25, 2.0, 0.05 m. These are fixed assumptions, not an
AERSURFACE calculation or site survey. Each station's locations, elevations,
wind-instrument height and time offset come from G:20. Random wind direction uses
AERMET's fixed EPA table (G:476); facility randomness is separate.

Reproduce all processing settings and constants:

```bash
nl -ba results/evidence/provenance/scenario-generator.py.txt | sed -n '20,26p;145,198p;450,476p'
```

### Preserved met-file hashes

These hashes were computed from the extracted bytes, independently of the raw
archive pins. X records their original instance membership and source paths.
Reproduce this entire table:

```bash
find convert/v1-src/met -type f -print0 | sort -z | xargs -0 sha256sum
```

| Preserved file | SHA-256 |
|---|---|
| `convert/v1-src/met/KABQ/KABQ.pfl` | `4ce9b833190f56df051fc95c45164fa056ac352bbd9caa8daa9cc7fe49cfce4f` |
| `convert/v1-src/met/KABQ/KABQ.sfc` | `36b3b6888617affbe1ce0bb9a59ab44d2a43121b7b071f07e72276aacc726426` |
| `convert/v1-src/met/KALB/KALB.pfl` | `8f4e449e44fe858f9cb8a4ac677987ca614f56ac98249ddb89059d4decd1d65c` |
| `convert/v1-src/met/KALB/KALB.sfc` | `42c6bb8194df683db8fadf0a4768ec6f415b5900daf21b2d50085352741ddfee` |
| `convert/v1-src/met/KAMA/KAMA.pfl` | `5dbae9111b28994c67d912fb9fafe501db04dd6587a717ac95fe8587179c906b` |
| `convert/v1-src/met/KAMA/KAMA.sfc` | `2d001268c5dcb0d19b877e69ee7e90ea4a75a6568f69d6546ef633d65bed20ba` |
| `convert/v1-src/met/KCRP/KCRP.pfl` | `a765112adfbe53912d357db04539e417a5a666efc44fe68c154022a5f9c1027a` |
| `convert/v1-src/met/KCRP/KCRP.sfc` | `d354b64ddf06d403274a28cecf339a51eaaf8afd72d7015d6c0bf08db02da5c2` |
| `convert/v1-src/met/KDDC/KDDC.pfl` | `9f2fac1c6e98d592ca39c7b59f1c6545d1096a68d4064e28902de46a53f38974` |
| `convert/v1-src/met/KDDC/KDDC.sfc` | `84c14c397c2fb8ad21a0e9d87ed46d190d41145eda0f96dcde9cc38361c9036b` |
| `convert/v1-src/met/KELP/KELP.pfl` | `8a2cef08e3bd7218f30392be0f6b86169d9b4ccf6c6a617bf8f58156fb48e737` |
| `convert/v1-src/met/KELP/KELP.sfc` | `6d68093ce40198bb6974893ff6ae5e30f19d70fa156d3c4f57f55983b6d91614` |
| `convert/v1-src/met/KGRB/KGRB.pfl` | `833a4537f02a3eb3f9bbd450b5d17e43cf163ed1c90f282a29df8e412a05963a` |
| `convert/v1-src/met/KGRB/KGRB.sfc` | `3e5ad19e21ecba274b91ce03876810b916da5f9be8c1be23c5cc47ed491cc50f` |
| `convert/v1-src/met/KIAD/KIAD.pfl` | `376c15652a5e3324f1fcd0aaefb8084397862d5bc7ba1b423539a074ccc9c52b` |
| `convert/v1-src/met/KIAD/KIAD.sfc` | `c276bc30875b9e3428fdd820f3202d23a499ddb5184342df9e09993e04c34f8e` |
| `convert/v1-src/met/KJAX/KJAX.pfl` | `637afb8aa1396cc4044e861524cb6ab6afc2bbf8e9291008da2a4228083331dc` |
| `convert/v1-src/met/KJAX/KJAX.sfc` | `8d080a4ac1e14db3e1cb596e77405b60a10c46986925998e9524f4074a58d608` |
| `convert/v1-src/met/KMSP/KMSP.pfl` | `9a420020eb6b920a298c62175391c0d9587995df117740f0c0a212bc7339a1e5` |
| `convert/v1-src/met/KMSP/KMSP.sfc` | `f472448a87fee23a99d70a195457bd5345425fd44e4e779149693a2af6f74c2b` |
| `convert/v1-src/met/KOMA/KOMA.pfl` | `438a0f5fd6e5fabd688c5b3c36a83783ad29c2dde6d9d6f203b134a040f4ff21` |
| `convert/v1-src/met/KOMA/KOMA.sfc` | `10666f4143f63a98e0333181b73b7334ed6450addffeca9e217dda4e0c4e9b16` |
| `convert/v1-src/met/KPIT/KPIT.pfl` | `63004d8bf9096a6a0ebb89e71110d6f19d9bdb6de79b7914824844bfb686653d` |
| `convert/v1-src/met/KPIT/KPIT.sfc` | `b9d7e406827eb3ab21796c4fc6178e6eba908dedc7572003fc1f60d160f74abd` |
| `convert/v1-src/met/KSHV/KSHV.pfl` | `f2dfa533294d63332794f59d08d77bf98d3990c94e46b1f8b838d3dce25a4155` |
| `convert/v1-src/met/KSHV/KSHV.sfc` | `888f40dc4f5006831833d41684975e9525be02056fcd70be9c03758e08db9357` |
| `convert/v1-src/met/KTOP/KTOP.pfl` | `fba709502622f264f26ca969a432d177c37052c356dd16b95cffd075fac9ae34` |
| `convert/v1-src/met/KTOP/KTOP.sfc` | `051f20e4aee8af89ac42d47de746d18841f266ce2e081553bcf7b9c03e783e4d` |
| `convert/v1-src/met/KTUS/KTUS.pfl` | `f35fd1a7265b22961559592d325ea01acf54898277034debd7813198c66ab2bd` |
| `convert/v1-src/met/KTUS/KTUS.sfc` | `c30bd0232dab8fdaa6f1153d6e82f1d26cc34583b3019e6f1e87d178761b1ad5` |

## EPA SCRAM model sources and guide

The preserved acquisition declaration E records these EPA URLs and archive/PDF pins. Rights: U.S. Government work,
`LicenseRef-US-Gov-Public-Domain` in E. The BPIP source identifies version 04274;
E describes its archive snapshot separately. AERMET was used to produce the met
files and is not bundled in the v2 native build.

Reproduce the upstream versions, URLs and acquisition hashes:

```bash
nl -ba results/evidence/provenance/runtime-source.yaml
head -40 envs/permit/core/native_build/model-source/bpipprm/Bpipprm.for
head -45 envs/permit/core/native_build/model-source/aermod/aermod.f
```

| Input | Version / recorded snapshot | Original file URL | SHA-256 of acquired file |
|---|---|---|---|
| AERMOD source | 26135; July 2026 memo | [aermod_source.zip](https://gaftp.epa.gov/Air/aqmg/SCRAM/models/preferred/aermod/aermod_source.zip) | `5092c1d68b77d9407c9f67d497b440a79d2ee746f9ed6515a63ad4a5b11cd8ed` |
| AERMET source | 26135 | [aermet_source.zip](https://gaftp.epa.gov/Air/aqmg/SCRAM/models/met/aermet/aermet_source.zip) | `11b0f7efaaa384fe20651d6553aed6c3604803f0a3e8bd103b925e88f7088686` |
| BPIPPRM source/examples | 04274; archive served 2026-10-06, source dated 2004-12-07 | [bpipprime.zip](https://gaftp.epa.gov/Air/aqmg/SCRAM/models/related/bpip/bpipprime.zip) | `a51a93fc8c306065830ded33772ad473fed9d10114dd5b317026c5616b21ed6f` |
| AERMOD User's Guide | EPA-454/B-26-001, July 2026 | [aermod_userguide.pdf](https://gaftp.epa.gov/Air/aqmg/SCRAM/models/preferred/aermod/aermod_userguide.pdf) | `8b747d154076635c2057bb939050fff196004312cceb8a8fc04688ee694a58e5` |

The extracted guide is `convert/v1-src/shared/docs/aermod_userguide.pdf`; validate
its acquired-file hash with `sha256sum convert/v1-src/shared/docs/aermod_userguide.pdf`.
The source-tree hashes below identify the files actually bundled for the verifier,
not the EPA ZIP bytes. Their lift origins are in the core README and build-script
headers. The build copies and instruments AERMOD in its build stage; it does not
modify the bundled originals.

### Compiler and snapshot pins

The original scenario generator used `gcc:13.4.0-bookworm`, Python
`3.12.13-bookworm` and `PyYAML==6.0.2` (C:1-5). Its compilation flags were:
AERMOD `-O2 -static -fbounds-check`; AERMET
`-O2 -static -fcheck=all -std=f2008 -ffree-form`; BPIP
`-O2 -static -std=legacy -fallow-argument-mismatch`, removing DOS SUB bytes from
BPIP's compilation copy (G:95-126). Source order came from EPA's batch files.
The source declaration
`results/evidence/provenance/scenario-source.yaml`:36-54 also
records seed `0`, runtime product `aermod-regulatory-runtime@0.1.0`, declared product
manifest digest `62cfc151ed05414fe0a0506d4c5d4ffc7a886adbca9c06bc47f82bb9f8be477f`,
and generator image digest
`sha256:603cc1850ca24e42b65b5d62fc7db57f9bc9ff3cc2d0d0f50b850ba72407fe71`.
The declared product digest is not a SHA-256 of the pretty-printed manifest file.
Reproduce with `nl -ba results/evidence/provenance/scenario-source.yaml`.
The base image tags themselves are recorded tags, not invented base-image digests.

The current v1-derived verifier build is pinned to
`debian:bookworm-20250224-slim@sha256:12c396bd585df7ec21d5679bb6a83d4878bc4415ce926c9e5ea6426d23c60bdc`
and apt snapshot `http://snapshot.debian.org/archive/debian/20250224T000000Z`.
It installs `gfortran` and Python from that snapshot; it does not explicitly pin
GCC 13. Its AERMOD compile flags add `-Wuninitialized` to
`-O2 -static -fbounds-check`, followed by static linking. BPIP uses
`-O2 -static -std=legacy` after stripping SUB only in scratch input.
The verifier applies `instrument.py` for per-year statistics, without the v1
interactive `screening_patch.py`. Build outputs have their own `bin/SHA256SUMS`.

Reproduce these pins and flags without building or launching native models:

```bash
nl -ba results/evidence/provenance/generator-container.txt
nl -ba results/evidence/provenance/scenario-generator.py.txt | sed -n '95,126p'
nl -ba envs/permit/core/native_build/Dockerfile
nl -ba envs/permit/core/native_build/build-models.sh
nl -ba envs/permit/core/native_build/build.sh
```

### Bundled source-file hashes

Reproduce file hashes and the two tree digests below. A tree digest is SHA-256 of
the UTF-8 concatenation of sorted lines `FILE_SHA256  PATH\n`, where `PATH` is
relative to `model-source/` and the trailing newline is included for each file.

```bash
uv run python - <<'PYTREE'
from pathlib import Path
import hashlib
base = Path('envs/permit/core/native_build/model-source')
for tree in sorted(base.iterdir()):
    rows = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(base).as_posix()}\n'
            for p in sorted(tree.rglob('*')) if p.is_file()]
    print(''.join(rows), end='')
    print(tree.name, 'tree SHA-256:', hashlib.sha256(''.join(rows).encode()).hexdigest())
PYTREE
```

| Bundled file | SHA-256 |
|---|---|
| `envs/permit/core/native_build/model-source/aermod/aermod.f` | `44c5c598fee18a44dbe8c16fc4b27e0ca373b21a7b6f0d873706f1c5b07a99bf` |
| `envs/permit/core/native_build/model-source/aermod/arise.f` | `90074ffa5ef128abbb6b09f8c8dad4645fa5605a7c315f09d5b7dcc87c3a260a` |
| `envs/permit/core/native_build/model-source/aermod/bline.f` | `47025c7656f719255c881107882e0cdcb6afd26f421771e18b249f663ae7f8fc` |
| `envs/permit/core/native_build/model-source/aermod/calc1.f` | `e08d51d6f2928379fb0514303141ed25269d359637bb0a6c7a4a4e0d862bb263` |
| `envs/permit/core/native_build/model-source/aermod/calc2.f` | `9103b2af8b1104eee8628e0dd7b02bea4e32165acc2d71aee8e981b33677d72b` |
| `envs/permit/core/native_build/model-source/aermod/coset.f` | `89afcc5f9995174014d42ea2ecde69de8368d53aab305ef595c5ff5bb1d1363a` |
| `envs/permit/core/native_build/model-source/aermod/evcalc.f` | `4a647482c90656624c885bd6fb8fd9903fda4fc364a042548eb9b02ebc419663` |
| `envs/permit/core/native_build/model-source/aermod/evoutput.f` | `424aae59cadd11d247c74a7e4ca3805a9355d0668ebd1d9c5b226bd6d749b13b` |
| `envs/permit/core/native_build/model-source/aermod/evset.f` | `cdc8dd1642215ada3377d5a23f8a54b2e38ab6e2a6e3652d2be78282893e54bd` |
| `envs/permit/core/native_build/model-source/aermod/grsm.f` | `f7d6d5fd0d0df453408aab3356c2fb16ded40f986f022b0fb6abb053543c8eb2` |
| `envs/permit/core/native_build/model-source/aermod/iblval.f` | `f3fc146021ffd4e3e378645f5104616aa1262f10dd9b456eb9b584221d7d967d` |
| `envs/permit/core/native_build/model-source/aermod/inpsum.f` | `5330b646c95838df3bf6bf46fb5cffbc9662c8b6fd6637869158d4485828ecbd` |
| `envs/permit/core/native_build/model-source/aermod/meset.f` | `77beaac764c360b4d7fa079ba7fba206597adf313ae8267a99a268dc6fa877da` |
| `envs/permit/core/native_build/model-source/aermod/metext.f` | `dd57eb349959c819733eeada410d2b61d04ce42969b61afba1e35bf8bfd779bc` |
| `envs/permit/core/native_build/model-source/aermod/modules.f` | `1b345773345d522119abec5eb912080db0c4407b3913c0254c9e213f39301802` |
| `envs/permit/core/native_build/model-source/aermod/ouset.f` | `373d4689869e88cee62963889517ef0a895473087c56ebe28c5b2d339b38f4a1` |
| `envs/permit/core/native_build/model-source/aermod/output.f` | `bb8d2ddcbab5cf11c8ac500ee3b4071c62173c01b0fc94f06343019f517ac51b` |
| `envs/permit/core/native_build/model-source/aermod/pitarea.f` | `836c492cdacd576c70544d99e83ce2c875653ede79cb0fd76b3232f92c1e8b05` |
| `envs/permit/core/native_build/model-source/aermod/prime.f` | `6d7aa489e81deee6e514cbdbb17ebc6042b4b0998450662fed3c4621590f5d97` |
| `envs/permit/core/native_build/model-source/aermod/prise.f` | `11338f7218adf920de5f5fb787df0dcfb20d92f367ba2d85868bb4024240cb06` |
| `envs/permit/core/native_build/model-source/aermod/reset.f` | `b03428350cd72331bffad629bebc15728fae5b18b7030a47439d8ad5ed48f6ef` |
| `envs/permit/core/native_build/model-source/aermod/rline.f` | `702ec274daf484990cda40388fdde9eb4e1d7c27b8f55c061a01f5db2702464d` |
| `envs/permit/core/native_build/model-source/aermod/setup.f` | `4e974f555c13acb933a07f957736bf40de06e559743b7224483cba2eec4e0adb` |
| `envs/permit/core/native_build/model-source/aermod/siggrid.f` | `ae87e094cba5fa1103258d12b0971ded3ad21ed315173eeb68ab0b92d2d89ad0` |
| `envs/permit/core/native_build/model-source/aermod/sigmas.f` | `b366889004b37720f83ca3f4371caaf24ce708a07ad0f32be43c10cc8385cc1c` |
| `envs/permit/core/native_build/model-source/aermod/soset.f` | `addd1e7eaae0f93e05cdb58566b0f6f0fbf012676c25af1ed2dbb60391733ac5` |
| `envs/permit/core/native_build/model-source/aermod/tempgrid.f` | `8e60fa47007f3849c2950bbabf0393173dbecfc89a7b7c4d22c78e0ccca260ed` |
| `envs/permit/core/native_build/model-source/aermod/uninam.f` | `e2bd90a5f67502999c51998d78bd891ec11e6d8d12c678a224b2a939c0a73360` |
| `envs/permit/core/native_build/model-source/aermod/windgrid.f` | `adb7676bdd1eacf9de650956d7550f7ed49e32f8f5a5249b0acab54d6bd3d5de` |
| `envs/permit/core/native_build/model-source/bpipprm/Bpipprm.for` | `0b8cc17ec7d28c1915076f4ca57e908680e6ec3fc0b695ed89c217a40be5bece` |

| Tree | SHA-256 of canonical file-hash lines |
|---|---|
| `aermod` | `2cebba7fb43e9d512591874e8c72cee92d38723cfeb2d72640aabf2c05f228a3` |
| `bpipprm` | `1e9ac20d92becd11cb27c69cbdc9c9e12c0cabe67a8195187522fc85b41772c7` |

## Synthetic inputs and selection

Facilities, costs, backgrounds and constant ozone are synthetic. No plant,
emissions inventory, cost quote, ambient background or ozone observation is real
permit data. G:378-388 seeds `random.Random` with the integer SHA-256 of
UTF-8 seed concatenated with ASCII index, chooses a station and archetype, and
tries candidates until the admission checks pass. Thus the retained distribution
is conditioned on admission; it is not an unfiltered sample of the ranges below.

Reproduce all draw ranges, geometry and admission constants:

```bash
nl -ba results/evidence/provenance/scenario-generator.py.txt | sed -n '26,49p;199,235p;378,457p'
```

| Input | How drawn / fixed | Source |
|---|---|---|
| Buildings | 1-4 axis-aligned rectangles, centers selected in order from (-90,-90), (90,-90), (-90,90), (90,90) m; width/depth uniform 25-65 m, height uniform 12-40 m, rounded to 2 decimals | G:214-219 |
| Sources | 2-6; buildings assigned cyclically; east-side offset uniform 3-12 m, north/south offset uniform -10 to 10 m; coordinates rounded to 2 decimals | G:221-230 |
| Stack bounds | minimum uniform 8 to min(25, building height) m, rounded to 2 decimals; maximum = BPIP GEP with 65 m floor | G:233, 415-420, 455 |
| In-stack NO2/NOx ratio | uniform 0.08-0.25, rounded to 4 decimals; retained as metadata in v2 | G:232 |
| Control cost | maximum uniform 120,000-500,000 USD/year per source; moderate = 0.55 × maximum, none = zero, rounded to cents | G:398-410 |
| Control efficiency | maximum NO2/SO2/PM25 = 0.95/0.98/0.995; moderate = 0.75 × each; none = zero | G:401-408 |
| Stack extension cost | one facility-wide price uniform 1,500-6,500 USD/(m·year), rounded to cents | G:398-399 |
| Background | each standard's limit × uniform 0.10-0.40, rounded to 6 decimals; PM25 annual secondary copied from primary | G:411-412 |
| Ozone | constant uniform 30-55 ppb, rounded to 3 decimals; v1 fallback, no hourly observations; unused by ARM2 in v2 | G:413, 442-443; `convert/v1_to_v2.py` |
| Terrain and surface | flat, rural; fixed assumed land-use triplet, no measured land cover | G:445, 453-454 |
| Receptors | square fence at ±250 m with 50 m spacing; 250 m grid to 1 km, 1 km grid inside 5 km, unique sorted points, elevation zero | G:199-210, 395 |
| Original admission | uncontrolled at minimum heights must fail a standard; all-maximum at GEP must pass every standard; redraw otherwise | G:424-433 |

The source archetype is chosen from sorted names. `small_refinery_mix` cycles
source types through gas boilers, process heaters and refinery mix. Each parameter
below is uniform within its interval and rounded to 6 decimals (G:27-34, 225-234).
NOx is expressed as NO2 mass before chemical conversion, not measured NO2.

| Source archetype | NOx as NO2 (g/s) | SO2 (g/s) | PM25 (g/s) | Temperature (K) | Exit velocity (m/s) | Diameter (m) |
|---|---|---|---|---|---|---|
| `gas_boilers` | 8-40 | 0.02-0.3 | 0.08-0.8 | 400-480 | 8-18 | 0.6-1.6 |
| `simple_cycle_turbine` | 20-100 | 0.05-0.8 | 0.1-1.5 | 650-800 | 15-28 | 1.5-3.5 |
| `process_heaters` | 10-60 | 0.2-4 | 0.3-2 | 430-600 | 8-20 | 0.7-2.0 |
| `kiln` | 25-100 | 3-40 | 2-12 | 400-550 | 10-22 | 1.2-3.0 |
| `small_refinery_mix` | 15-80 | 2-25 | 0.5-6 | 400-600 | 8-22 | 0.8-2.5 |

The benchmark thresholds and statistical forms were fixed by G:35-49 (which cites
the [EPA NAAQS table](https://www.epa.gov/criteria-air-pollutants/naaqs-table) in
G:448), then preserved in each task. [CONTRACT.md](CONTRACT.md) describes the exact
benchmark rules; this ledger makes no claim that future regulations stay unchanged.
V2 changes the NO2 tier to ARM2, discretizes heights and rechecks admission; it does
not regenerate facilities (`convert/v1_to_v2.py`).

## Other external inputs

The OpenEnv adapter follows PortSim's structure in the FineEnvs repository
(`07-simulation-environments/portsim-v1/`). The root README follows its section
order. Physical grading comes from the published PermitForge v1 grader; exact
function origins are listed in [the core README](envs/permit/core/README.md).

The v2 reward is the saved fraction: feasible annual USD saved divided by the
native-verified draft cost, with zero for a failed or missing submission. It is
implemented in `envs/permit/core/permit_core/reward.py`; raw savings and exact
cost retain the published v1 grader's provenance. This normalization is local
PermitForge code, with no external reward helper or source snapshot.
Reproduce with `uv run --directory envs/permit/core --no-sync pytest tests/test_reward.py -q`.
