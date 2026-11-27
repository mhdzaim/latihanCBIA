from pathlib import Path

import streamlit as st
import pandas as pd
import pickle
import altair as alt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

st.title("Dashboard Analitik DJPb")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

data_path = Path(__file__).parent / "data" / "02_realisasi_anggaran_klasifikasi.csv"
model_path = Path(__file__).parent / "models" / "Best_model.pkcls"
data = pd.read_csv(data_path)

@st.cache_resource
def load_classification_model():
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_resource
def train_regression_model():
    fitur_numerik = [
        "pagu_miliar",
        "jumlah_pegawai",
        "jumlah_spm",
        "revisi_dipa",
        "realisasi_tw1_persen",
        "realisasi_tw2_persen",
        "deviasi_rpd_persen",
    ]
    fitur_kategorikal = ["tipe_satker", "jenis_belanja_utama"]
    fitur = fitur_numerik + fitur_kategorikal
    data_latih = data[fitur]
    target = data["realisasi_tw3_persen"]

    preprocessing = ColumnTransformer(
        [
            ("numerik", "passthrough", fitur_numerik),
            (
                "kategori",
                OneHotEncoder(handle_unknown="ignore"),
                fitur_kategorikal,
            ),
        ]
    )
    model = make_pipeline(preprocessing, Ridge())
    validasi = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_validasi = -cross_val_score(
        model,
        data_latih,
        target,
        cv=validasi,
        scoring="neg_mean_absolute_error",
    ).mean()
    model.fit(data_latih, target)
    return model, float(mae_validasi), fitur


tab_prediksi, tab_visualisasi, tab_dataset, tab_peta = st.tabs(
    ["Menu Prediksi", "Visualisasi", "Data/Dataset", "Peta Satker"]
)

with tab_prediksi:
    st.header("Prediksi Realisasi Anggaran")

    if opsi == "Regresi":
        st.caption(
            "Prediksi persentase realisasi TW3 berdasarkan karakteristik satker "
            "dan data realisasi TW1–TW2. Model dilatih dari dataset yang tersedia."
        )
        with st.form("form_prediksi_regresi"):
            kolom_kiri, kolom_kanan = st.columns(2)
            with kolom_kiri:
                pagu_miliar = st.number_input(
                    "Pagu (miliar rupiah)",
                    min_value=float(data["pagu_miliar"].min()),
                    max_value=float(data["pagu_miliar"].max()),
                    value=float(data["pagu_miliar"].median()),
                    step=0.1,
                )
                jumlah_pegawai = st.number_input(
                    "Jumlah Pegawai",
                    min_value=int(data["jumlah_pegawai"].min()),
                    max_value=int(data["jumlah_pegawai"].max()),
                    value=int(data["jumlah_pegawai"].median()),
                    step=1,
                )
                jumlah_spm = st.number_input(
                    "Jumlah SPM",
                    min_value=int(data["jumlah_spm"].min()),
                    max_value=int(data["jumlah_spm"].max()),
                    value=int(data["jumlah_spm"].median()),
                    step=1,
                )
                revisi_dipa = st.number_input(
                    "Revisi DIPA",
                    min_value=int(data["revisi_dipa"].min()),
                    max_value=int(data["revisi_dipa"].max()),
                    value=int(data["revisi_dipa"].median()),
                    step=1,
                )
                deviasi_rpd_persen = st.number_input(
                    "Deviasi RPD (%)",
                    min_value=float(data["deviasi_rpd_persen"].min()),
                    max_value=float(data["deviasi_rpd_persen"].max()),
                    value=float(data["deviasi_rpd_persen"].median()),
                    step=0.1,
                )
            with kolom_kanan:
                realisasi_tw1_persen = st.number_input(
                    "Realisasi TW1 (%)",
                    min_value=float(data["realisasi_tw1_persen"].min()),
                    max_value=float(data["realisasi_tw1_persen"].max()),
                    value=float(data["realisasi_tw1_persen"].median()),
                    step=0.1,
                )
                realisasi_tw2_persen = st.number_input(
                    "Realisasi TW2 (%)",
                    min_value=float(data["realisasi_tw2_persen"].min()),
                    max_value=float(data["realisasi_tw2_persen"].max()),
                    value=float(data["realisasi_tw2_persen"].median()),
                    step=0.1,
                )
                tipe_satker = st.selectbox(
                    "Tipe Satker",
                    sorted(data["tipe_satker"].dropna().unique()),
                    key="regresi_tipe_satker",
                )
                jenis_belanja = st.selectbox(
                    "Jenis Belanja Utama",
                    sorted(data["jenis_belanja_utama"].dropna().unique()),
                    key="regresi_jenis_belanja",
                )
            prediksi_regresi_diajukan = st.form_submit_button("Prediksi Realisasi TW3")

        if prediksi_regresi_diajukan:
            model_regresi, mae_validasi, fitur_regresi = train_regression_model()
            input_regresi = pd.DataFrame(
                [
                    {
                        "pagu_miliar": pagu_miliar,
                        "jumlah_pegawai": jumlah_pegawai,
                        "jumlah_spm": jumlah_spm,
                        "revisi_dipa": revisi_dipa,
                        "realisasi_tw1_persen": realisasi_tw1_persen,
                        "realisasi_tw2_persen": realisasi_tw2_persen,
                        "deviasi_rpd_persen": deviasi_rpd_persen,
                        "tipe_satker": tipe_satker,
                        "jenis_belanja_utama": jenis_belanja,
                    }
                ],
                columns=fitur_regresi,
            )
            prediksi_realisasi = float(model_regresi.predict(input_regresi)[0])
            st.metric("Prediksi Realisasi TW3", f"{prediksi_realisasi:.2f}%")
            st.caption(
                f"MAE validasi silang 5-fold: {mae_validasi:.2f} poin persentase "
                "(rata-rata selisih absolut pada data uji)."
            )
    else:
        st.caption(
            "Masukkan data satker untuk memprediksi apakah realisasi anggaran mencapai 95%."
        )

        with st.form("form_prediksi_klasifikasi"):
            kolom_kiri, kolom_kanan = st.columns(2)

            with kolom_kiri:
                jumlah_spm = st.number_input(
                    "Jumlah SPM",
                    min_value=0,
                    value=int(data["jumlah_spm"].median()),
                    step=1,
                )
                revisi_dipa = st.number_input(
                    "Revisi DIPA",
                    min_value=0,
                    value=int(data["revisi_dipa"].median()),
                    step=1,
                )
                skor_ikpa = st.number_input(
                    "Skor IKPA",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(data["skor_ikpa"].median()),
                    step=0.1,
                )

            with kolom_kanan:
                deviasi_rpd_persen = st.number_input(
                    "Deviasi RPD (%)",
                    min_value=0.0,
                    value=float(data["deviasi_rpd_persen"].median()),
                    step=0.1,
                )
                tipe_satker = st.selectbox(
                    "Tipe Satker",
                    options=sorted(data["tipe_satker"].dropna().unique()),
                )

            prediksi_diajukan = st.form_submit_button("Prediksi", type="primary")

        if prediksi_diajukan:
            try:
                from Orange.data import Table
            except ModuleNotFoundError as error:
                if error.name != "Orange":
                    raise
                st.error(
                    "Dependensi Orange3 belum tersedia. Install dengan "
                    "`pip install Orange3`, lalu jalankan ulang aplikasi."
                )
            else:
                model = load_classification_model()
                domain = model.original_domain
                input_fitur = {
                    "jumlah_spm": jumlah_spm,
                    "revisi_dipa": revisi_dipa,
                    "deviasi_rpd_persen": deviasi_rpd_persen,
                    "skor_ikpa": skor_ikpa,
                }
                input_fitur.update(
                    {
                        f"tipe_satker={kategori}": float(tipe_satker == kategori)
                        for kategori in data["tipe_satker"].dropna().unique()
                    }
                )
                nilai_baris = [
                    input_fitur[variable.name] for variable in domain.attributes
                ]
                nilai_baris.extend([float("nan")] * len(domain.class_vars))
                nilai_baris.extend([""] * len(domain.metas))
                data_prediksi = Table.from_list(domain, [nilai_baris])
                hasil = model(data_prediksi)
                probabilitas = model(data_prediksi, model.Probs)[0]
                kelas_prediksi = domain.class_var.values[int(hasil[0])]
                keyakinan = float(probabilitas[int(hasil[0])])
                pesan_prediksi = (
                    f"Prediksi realisasi mencapai 95%: **{kelas_prediksi}** "
                    f"(keyakinan {keyakinan:.1%})"
                )
                if kelas_prediksi == "Ya":
                    st.success(pesan_prediksi)
                else:
                    st.warning(pesan_prediksi)

with tab_dataset:
    st.header("Data / Dataset")

    semua_kementerian = ["Semua", *sorted(data["nama_kementerian"].dropna().unique())]
    semua_provinsi = ["Semua", *sorted(data["provinsi"].dropna().unique())]
    semua_tipe_satker = ["Semua", *sorted(data["tipe_satker"].dropna().unique())]
    semua_status = [
        "Semua",
        *sorted(data["realisasi_tercapai_95persen"].dropna().unique()),
    ]

    kolom_filter_1, kolom_filter_2, kolom_filter_3 = st.columns(3)
    with kolom_filter_1:
        filter_kementerian = st.selectbox(
            "Kementerian",
            semua_kementerian,
            key="filter_dataset_kementerian",
        )
        filter_tipe_satker = st.selectbox(
            "Tipe Satker",
            semua_tipe_satker,
            key="filter_dataset_tipe_satker",
        )
    with kolom_filter_2:
        filter_provinsi = st.selectbox(
            "Provinsi",
            semua_provinsi,
            key="filter_dataset_provinsi",
        )
        filter_status = st.selectbox(
            "Realisasi mencapai 95%",
            semua_status,
            key="filter_dataset_status",
        )
    with kolom_filter_3:
        rentang_ikpa = st.slider(
            "Rentang Skor IKPA",
            min_value=70.0,
            max_value=100.0,
            value=(
                float(data["skor_ikpa"].min()),
                float(data["skor_ikpa"].max()),
            ),
            step=0.1,
            key="filter_dataset_ikpa",
        )
        cari_satker = st.text_input(
            "Cari Kode Satker",
            key="filter_dataset_kode_satker",
        )

    data_tampil = data[
        data["skor_ikpa"].between(rentang_ikpa[0], rentang_ikpa[1])
    ]
    if filter_kementerian != "Semua":
        data_tampil = data_tampil[
            data_tampil["nama_kementerian"] == filter_kementerian
        ]
    if filter_provinsi != "Semua":
        data_tampil = data_tampil[data_tampil["provinsi"] == filter_provinsi]
    if filter_tipe_satker != "Semua":
        data_tampil = data_tampil[data_tampil["tipe_satker"] == filter_tipe_satker]
    if filter_status != "Semua":
        data_tampil = data_tampil[
            data_tampil["realisasi_tercapai_95persen"] == filter_status
        ]
    if cari_satker:
        data_tampil = data_tampil[
            data_tampil["kode_satker"].str.contains(
                cari_satker,
                case=False,
                na=False,
                regex=False,
            )
        ]

    st.caption(
        f"Menampilkan {len(data_tampil)} dari {len(data)} baris "
        f"({len(data.columns)} kolom)."
    )
    st.dataframe(data_tampil, width="stretch", hide_index=True)

with tab_peta:
    st.header("Peta Sebaran Satker")
    st.caption(
        "Lokasi titik menunjukkan pusat wilayah provinsi secara indikatif, "
        "bukan koordinat persis satker."
    )

    koordinat_provinsi = {
        "Aceh": (4.6951, 96.7494),
        "Bali": (-8.4095, 115.1889),
        "DKI Jakarta": (-6.2088, 106.8456),
        "Jawa Barat": (-6.9175, 107.6191),
        "Jawa Tengah": (-7.1510, 110.1403),
        "Jawa Timur": (-7.5361, 112.2384),
        "Kalimantan Timur": (0.5387, 116.4194),
        "Lampung": (-4.5586, 105.4068),
        "NTB": (-8.6529, 117.3616),
        "NTT": (-8.6574, 121.0794),
        "Papua": (-4.2699, 138.0804),
        "Riau": (0.2933, 101.7068),
        "Sulawesi Selatan": (-3.6688, 119.9741),
        "Sumatera Barat": (-0.7399, 100.8000),
        "Sumatera Utara": (2.1154, 99.5451),
    }

    kementerian_peta = st.selectbox(
        "Kementerian",
        ["Semua", *sorted(data["nama_kementerian"].dropna().unique())],
        key="peta_kementerian",
    )
    kolom_filter_peta_1, kolom_filter_peta_2, kolom_filter_peta_3 = st.columns(3)
    with kolom_filter_peta_1:
        provinsi_peta = st.multiselect(
            "Provinsi",
            sorted(data["provinsi"].dropna().unique()),
            default=sorted(data["provinsi"].dropna().unique()),
            key="peta_provinsi",
        )
        tipe_satker_peta = st.selectbox(
            "Tipe Satker",
            ["Semua", *sorted(data["tipe_satker"].dropna().unique())],
            key="peta_tipe_satker",
        )
    with kolom_filter_peta_2:
        jenis_belanja_peta = st.selectbox(
            "Jenis Belanja Utama",
            ["Semua", *sorted(data["jenis_belanja_utama"].dropna().unique())],
            key="peta_jenis_belanja",
        )
        status_peta = st.selectbox(
            "Realisasi Mencapai 95%",
            ["Semua", *sorted(data["realisasi_tercapai_95persen"].dropna().unique())],
            key="peta_status",
        )
    with kolom_filter_peta_3:
        rentang_ikpa_peta = st.slider(
            "Rentang Skor IKPA",
            min_value=70.0,
            max_value=100.0,
            value=(70.0, 100.0),
            step=0.1,
            key="peta_rentang_ikpa",
        )

    mask_peta = data["provinsi"].isin(provinsi_peta) & data["skor_ikpa"].between(
        rentang_ikpa_peta[0], rentang_ikpa_peta[1]
    )
    if kementerian_peta != "Semua":
        mask_peta &= data["nama_kementerian"] == kementerian_peta
    if tipe_satker_peta != "Semua":
        mask_peta &= data["tipe_satker"] == tipe_satker_peta
    if jenis_belanja_peta != "Semua":
        mask_peta &= data["jenis_belanja_utama"] == jenis_belanja_peta
    if status_peta != "Semua":
        mask_peta &= data["realisasi_tercapai_95persen"] == status_peta

    data_peta = data.loc[mask_peta]
    if data_peta.empty:
        st.info("Tidak ada satker yang sesuai dengan filter yang dipilih.")
    else:
        ringkasan_peta = (
            data_peta.groupby("provinsi", as_index=False)
            .agg(
                jumlah_satker=("kode_satker", "count"),
                rata_rata_ikpa=("skor_ikpa", "mean"),
                persentase_tercapai=(
                    "realisasi_tercapai_95persen",
                    lambda status: status.eq("Ya").mean() * 100,
                ),
            )
        )
        provinsi_tanpa_koordinat = sorted(
            set(ringkasan_peta["provinsi"]) - set(koordinat_provinsi)
        )
        if provinsi_tanpa_koordinat:
            st.error(
                "Koordinat peta belum tersedia untuk: "
                + ", ".join(provinsi_tanpa_koordinat)
            )
        else:
            ringkasan_peta["latitude"] = ringkasan_peta["provinsi"].map(
                lambda provinsi: koordinat_provinsi[provinsi][0]
            )
            ringkasan_peta["longitude"] = ringkasan_peta["provinsi"].map(
                lambda provinsi: koordinat_provinsi[provinsi][1]
            )
            ringkasan_peta["ukuran_marker"] = (
                10000 + ringkasan_peta["jumlah_satker"] * 500
            )
            ringkasan_peta["warna"] = ringkasan_peta[
                "persentase_tercapai"
            ].map(lambda persentase: "#008000" if persentase >= 50 else "#d62728")

            st.caption(
                f"Menampilkan {len(data_peta)} satker di "
                f"{len(ringkasan_peta)} provinsi. Marker lebih besar menunjukkan "
                "jumlah satker lebih banyak; hijau berarti mayoritas mencapai 95%."
            )
            st.map(
                ringkasan_peta,
                latitude="latitude",
                longitude="longitude",
                size="ukuran_marker",
                color="warna",
                height=600,
                alt="Peta jumlah satker dan rata-rata skor IKPA per provinsi",
            )
            st.dataframe(
                ringkasan_peta[
                    [
                        "provinsi",
                        "jumlah_satker",
                        "rata_rata_ikpa",
                        "persentase_tercapai",
                    ]
                ].rename(
                    columns={
                        "provinsi": "Provinsi",
                        "jumlah_satker": "Jumlah Satker",
                        "rata_rata_ikpa": "Rata-rata IKPA",
                        "persentase_tercapai": "Pencapaian 95% (%)",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

with tab_visualisasi:
    st.header("Visualisasi Bubble Plot")
    st.markdown(
        "Grafik bubble menunjukkan skor IKPA terhadap deviasi RPD, dengan warna berdasarkan kolom `realisasi_tercapai_95persen`."
    )

    chart = (
        alt.Chart(data)
        .mark_circle(opacity=0.7, size=100)
        .encode(
            x=alt.X(
                "skor_ikpa",
                title="Skor IKPA",
                scale=alt.Scale(domain=[70, 100]),
            ),
            y=alt.Y("deviasi_rpd_persen", title="Deviasi RPD (%)"),
            color=alt.Color(
                "realisasi_tercapai_95persen:N",
                title="Realisasi Tercapai 95%",
                scale=alt.Scale(domain=["Ya", "Tidak"], range=["green", "red"]),
            ),
            tooltip=[
                "kode_satker",
                "tipe_satker",
                "skor_ikpa",
                "deviasi_rpd_persen",
                "realisasi_tercapai_95persen",
            ],
        )
        .properties(width=900, height=550)
        .interactive()
    )
    st.altair_chart(chart, width=900)

    st.subheader("Histogram Skor IKPA")
    histogram = (
        alt.Chart(data)
        .mark_bar(color="green")
        .encode(
            x=alt.X(
                "skor_ikpa:Q",
                bin=alt.Bin(step=2, extent=[70, 100]),
                title="Skor IKPA",
                scale=alt.Scale(domain=[70, 100]),
                axis=alt.Axis(values=list(range(70, 101, 2))),
            ),
            y=alt.Y("count():Q", title="Jumlah satker"),
            tooltip=[alt.Tooltip("count():Q", title="Jumlah satker")],
        )
        .properties(width=900, height=350)
    )
    st.altair_chart(histogram, width=900)

    st.subheader("Analisis Tambahan")
    skor_per_tipe = (
        data.groupby("tipe_satker", as_index=False)["skor_ikpa"]
        .mean()
        .rename(columns={"skor_ikpa": "rata_rata_ikpa"})
    )
    bar_chart = (
        alt.Chart(skor_per_tipe)
        .mark_bar(color="seagreen")
        .encode(
            x=alt.X(
                "rata_rata_ikpa:Q",
                title="Rata-rata Skor IKPA",
                scale=alt.Scale(domain=[70, 100]),
            ),
            y=alt.Y(
                "tipe_satker:N",
                title="Tipe Satker",
                sort="-x",
                axis=alt.Axis(labelLimit=180, labelFontSize=12),
            ),
            tooltip=[
                alt.Tooltip("tipe_satker:N", title="Tipe Satker"),
                alt.Tooltip("rata_rata_ikpa:Q", title="Rata-rata IKPA", format=".2f"),
            ],
        )
        .properties(width=900, height=360)
    )
    st.altair_chart(bar_chart, width=900)

    kolom_triwulan = {
        "TW1": "realisasi_tw1_persen",
        "TW2": "realisasi_tw2_persen",
        "TW3": "realisasi_tw3_persen",
    }
    rata_realisasi = pd.DataFrame(
        {
            "triwulan": list(kolom_triwulan),
            "rata_rata_realisasi": [
                data[kolom].mean() for kolom in kolom_triwulan.values()
            ],
        }
    )
    line_chart = (
        alt.Chart(rata_realisasi)
        .mark_line(point=True, color="seagreen")
        .encode(
            x=alt.X("triwulan:N", title="Triwulan", sort=list(kolom_triwulan)),
            y=alt.Y(
                "rata_rata_realisasi:Q",
                title="Rata-rata Realisasi (%)",
            ),
            tooltip=[
                alt.Tooltip("triwulan:N", title="Triwulan"),
                alt.Tooltip(
                    "rata_rata_realisasi:Q",
                    title="Rata-rata realisasi (%)",
                    format=".2f",
                ),
            ],
        )
        .properties(width=900, height=320)
    )
    st.altair_chart(line_chart, width=900)

    st.subheader("Sebaran Deviasi RPD berdasarkan Pencapaian")
    box_chart = (
        alt.Chart(data)
        .mark_boxplot()
        .encode(
            x=alt.X(
                "realisasi_tercapai_95persen:N",
                title="Realisasi Tercapai 95%",
                sort=["Ya", "Tidak"],
            ),
            y=alt.Y("deviasi_rpd_persen:Q", title="Deviasi RPD (%)"),
            color=alt.Color(
                "realisasi_tercapai_95persen:N",
                scale=alt.Scale(domain=["Ya", "Tidak"], range=["green", "red"]),
                legend=None,
            ),
        )
        .properties(width=900, height=350)
    )
    st.altair_chart(box_chart, width=900)