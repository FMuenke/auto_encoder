import os
import shutil
from auto_encoder.data_set import DataSet

def main():
    cls_groups = {
        "Halteverbot": [
            'Absolutes_Halteverbot_outdated',
            'Absolutes_Haltverbot_Mitte_Rechtsaufstellung_283-30',
            'Absolutes_Haltverbot_Anfang_Rechtsaufstellung_283-10',
            'Absolutes_Haltverbot_Ende_Rechtsaufstellung_283-20',
            'Absolutes_Haltverbot_Ende_Linksaufstellung_283-11',
            'Absolutes_Haltverbot_Anfang_Linksaufstellung_283-21',
            'Absolutes_Haltverbot_283',
            'Absolutes_Haltverbot_Mitte_Linksaufstellung_283-31',
            'Eingeschraenktes_Halteverbot_outdated',
            'Eingeschraenktes_Haltverbot_286',
            'Eingeschraenktes_Haltverbot_Mitte_Rechtsaufstellung_Verkehrsschild_286-30',
            'Eingeschraenktes_Haltverbot_Ende_Linksaufstellung_286-11',
            'Eingeschraenktes_Haltverbot_Anfang_Rechtsaufstellung_286-10',
            'Eingeschraenktes_Haltverbot_Anfang_Linksaufstellung_286-21',
            'Eingeschraenktes_Haltverbot_Ende_Rechtsaufstellung_286-20',
            'Eingeschraenktes Haltverbot_Mitte_Linksaufstellung_286-31',
            'Beginn_Halteverbot_zone_290.1'
        ],
        "Parken": [
            'Parken_outdated',
            'Parken_Ende_Rechtsaufstellung_oder_Parken_Anfang_Linksaufstellung_314-20',
            'Parken_314',
            'Parken_Mitte_Rechts_oder_Linksaufstellung_314-30',
            'Parkhaus_Parkgarage_314-50',
            'Parken_auf_Gehwegen_315',
            'Parken Anfang_Rechtsaufstellung_oder_Parken_Ende_Linksaufstellung_314-10',
            'Beginn_einer_Parkraumbewirtschaftungszone_314.1'
        ],
        "Ende_der_zulaessigen_Hoechstgeschwindigkeit": [
            'Ende_des_Ueberholverbotes_fuer_Kraftfahrzeuge_mit_einer_zulaessigen_Gesamtmasse_ueber_3.5t_einschliesslich_ihrer_Anhaenger_und_Zugmaschinen_ausgenommen_PKW_und_Kraftomnibusse_281',
            'Ende_der_zulaessigen_Hoechstgeschwindigkeit_30_km_h_278-30',
            'Ende_der_zulaessigen_Hoechstgeschwindigkeit_50_km_h_278-50',
            'Ende_der_zulaessigen_Hoechstgeschwindigkeit_70_km_h_278-70',
            'Ende_der_zulaessigen_Hoechstgeschwindigkeit_80_km_h_278-80',
            'Ende_saemtlicher_streckenbezogener_Geschwindigkeitsbeschraenkungen_und_Ueberholverbote_282',
            'Ende_des_Ueberholverbotes_fuer_Kraftfahrzeuge_aller_Art_280'
        ],
        "Vorgeschriebene_Fahrtrichtung": [
            'Kreisverkehr_215',
            'Vorgeschriebene_Vorbeifahrt_rechts_222',
            'Vorgeschriebene_Fahrtrichtung_geradeaus_oder_rechts_214',
            'Vorgeschriebene Fahrtrichtung_geradeaus_oder_links_214-10',
            'Vorgeschriebene_Fahrtrichtung_rechts_209',
            'Vorgeschriebene_Fahrtrichtung_links_oder_rechts_214-30',
            'Vorgeschriebene_Vorbeifahrt_links_vorbei_222-10',
            'Vorgeschriebene_Fahrtrichtung_geradeaus_209-30',
            'Vorgeschriebene_Fahrtrichtung_hier_links_211-10',
            'Vorgeschriebene_Fahrtrichtung links_209-10',
            'Vorgeschriebene_Fahrtrichtung_hier_rechts_211'
        ],
        "Vorfahrtstrasse": [
            'Vorfahrtstrasse_306',
            'Ende_der_Vorfahrtstrasse_307'
        ],

        "Doppelkurve": [
            'Doppelkurve_zunaechst_rechts_105-20',
             'Doppelkurve_zunaechst_links_105-10'
        ],
        "Einbahnstrasse": [
            'Einbahnstrasse_rechts_220-20',
            'Einbahnstrasse_links_220-10'
        ],
        "Verengte_Fahrbahn": [
            'Verengte_Fahrbahn_120',
            'Einseitig_verengte_Fahrbahn_Verengung_rechts_121-10',
            'Einseitig_verengte_Fahrbahn_Verengung_links_121-20',
        ],
        "Fussgaengerueberweg_Blau": [
            'Fussgaengerueberweg_Aufstellung_rechts_350-10',
            'Fussgaengerueberweg_Aufstellung_links_350-20'
        ],
        "Fussgaengerueberweg_Rot": [
            'Fussgaengerueberweg_Aufstellung_rechts_101-11',
            'Fussgaengerueberweg_Aufstellung_links_101-21'
        ],
        "Gemeinsamer_Geh_Radweg": [
            'Gemeinsamer_Geh_Radweg_240',
            'Gemeinsamer_Geh_Radweg_Fahrrad_oben',
        ],
        "Getrennter_Geh_Radweg": [
            'Getrennter_Rad_und_Gehweg_Radweg_rechts_241-31',
            'Getrennter_Rad_und_Gehweg_Radweg_links_241-30'
        ],
        "Kinder": [
            'Kinder_Aufstellung_rechts_136-10',
            'Kinder_Aufstellung_links_136-20'
        ],
        "Kurve": [
            'Kurve_rechts_103-20',
            'Kurve_links_103-10'
        ],
        "Ortstafel": [
            'Ortstafel_Vorderseite_310',
            'Ortstafel_Rueckseite_weiss_gelb',
            'Ortstafel_Rueckseite_ohne_naechste_Stadt',
            'Ortstafel_Rueckseite_311'
        ],
        "Radverkehr": [
            'Radverkehr_Aufstellung_rechts_138-10',
            'Radverkehr_Aufstellung_links_138-20'
        ],
        "Sackgasse": [
            'Sackgasse_357',
            'Sackgasse_fuer_Fussgaenger_durchlaessig_357-51',
            'Sackgasse_fuer_Radverkehr_und_Fussgaenger_durchlaessig_357-50'
        ],
        "Wildwechsel": [
            'Wildwechsel_Aufstellung_rechts_142-10',
            'Wildwechsel_Aufstellung_links_142-20'
        ],
        "Leitpfosten": [
            'Leitpfosten_rechts_620-40',
            'Leitpfosten_links_620-41'
        ],
        "Verbot_fuer_Fahrzeuge_und_Zuege_ueber_angegebene_tatsaechliche_Laenge_266": [
            "10m", "12m", "16m", "unsorted"
        ],
        "Verbot_fuer_Fahrzeuge_ueber_angegebene_tatsaechliche_Hoehe_265": [
            "1.8m", "1.9m", "2.1m", "2.3m", "2.8m", "2.15m", "2.70m", "2m", "3.1m", "3.4m", "3.6m", "3.7m", "3.8m",
            "3.9m", "3.65m", "3m", "4.2m", "4.3m", "4m"
        ],
        "Verbot_fuer_Fahrzeuge_ueber_angegebenes_tatsaechliches_Gewicht_262": [
            "2.5t", "2.8t", "3.5t", "3t", "5.5t", "5t", "6.5t", "6t", "7.5t", "9t", "10.5t", "10t", "12t",
            "15t", "16t", "18t", "24t", "26t", "30t", "40t"
        ]
    }

    known_classes_not_grouped = ['Ende_einer_Tempo_30_Zone_274.2', 'Fussgaenger_Aufstellung_rechts_133-10',
                                 'Ende_Halteverbot_zone_290.2',
                                 'Ueberholverbot_fuer_Kraftfahrzeuge_aller_Art_276',
                                 'Allgemeine_Gefahrenstelle_101', 'Radweg_nicht_StVO',
                                 'Beginn_einer_Tempo_30_Zone_274.1', 'Verbot_der_Einfahrt_267', 'Parkscheinautomat',
                                 'Zulaessige_Hoechstgeschwindigkeit_50_km_h_274-50',
                                 'Verbot_fuer_Fahrzeuge_und_Zuege_ueber_angegebene_tatsaechliche_Laenge_266',
                                 'Zulaessige_Hoechstgeschwindigkeit_70_km_h_274-70', 'Gehweg_239',
                                 'Zulaessige_Hoechstgeschwindigkeit_80_km_h_274-80', 'Wandererparkplatz_317',
                                 'Kreuzung_oder_Einmuendung_102', 'Zulaessige_Hoechstgeschwindigkeit_20_km_h_274-20',
                                 'Lichtzeichenanlage_131', 'Verbot_fuer_Fussgaenger_259', 'Vorfahrt_301',
                                 'Ende_verkehrsberuhigter_Bereichs_325.2',
                                 'Schleudergefahr_bei_Naesse_oder_Schmutz_114',
                                 'Verbot_fuer_Fahrzeuge_ueber_angegebene_tatsaechliche_Hoehe_265',
                                 'Ueberholverbot_fuer_Kraftfahrzeuge_mit_einer_zulaessigen_Gesamtmasse_ueber_3,5_t_einschliesslich_ihrer_Anhaenger_und_Zugmaschinen_ausgenommen_PKW_und_Kraftomnibusse_277',
                                 'Verbot_fuer_Kraftfahrzeuge_mit_einer_zulaessigen_Gesamtmasse_ueber_3,5_t_einschliesslich_ihrer_Anhaenger_und_Zugmaschinen_ausgenommen_sind_PKW_und_Kraftomnibusse_253',
                                 'Vorrang_des_Gegenverkehrs_208', 'Arbeitsstelle_Baustelle_123',
                                 'Halt_Vorfahrt_gewaehren_206',
                                 'Verbot_fuer_Fahrzeuge_ueber_angegebenes_tatsaechliches_Gewicht_262',
                                 'Zulaessige_Hoechstgeschwindigkeit_10_km_h_274-10',
                                 'Zulaessige_Hoechstgeschwindigkeit_30_km_h_274-30',
                                 'Beginn_verkehrsberuhigter_Bereich_325.1', 'Bahnuebergang_151', 'Gegenverkehr_125',
                                 'Radweg_237', 'Beginn_einer_Tempo_20_Zone_274.1-20', 'Vorfahrt_gewaehren_205',
                                 'Schnee_oder_Eisglaette_101-51', 'Zulaessige_Hoechstgeschwindigkeit_120_km_h_274-120',
                                 'Zulaessige_Hoechstgeschwindigkeit_100_km_h_274-100', 'Unebene_Fahrbahn_112',
                                 'Zelt_und_Wohnwagenplatz_365-60',
                                 'Verbot_fuer_Kraftraeder_auch_mit_Beiwagen_Kleinkraftraeder_Mofas_sowie_fuer_Kraftwagen_und_sonstige_mehrspurige_Kfz_260',
                                 'Verbot_fuer_Fahrzeuge_aller_Art_250',
                                 'Zulaessige_Hoechstgeschwindigkeit_40_km_h_274-40',
                                 'Zulaessige_Hoechstgeschwindigkeit_60_km_h_274-60']

    for c in known_classes_not_grouped:
        if c in cls_groups:
            continue
        cls_groups[c] = [c]

    class_mapping = {c: i for i, c in enumerate(cls_groups)}
    mapping_to_groups = {}
    for grp in cls_groups:
        for c in cls_groups[grp]:
            mapping_to_groups[c] = grp

    path_to_data = "/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/datasets/Traffic_Sign_Dataset_vialytics_and_GTSRB_2022_07_11/test/unknown"
    path_dst = "/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/datasets/TS-Data/test/unknown"
    ds = DataSet(path_to_data)
    ds.load()
    for img in ds.get_data():
        shutil.copy(img.path, os.path.join(path_dst, img.name))

if __name__ == "__main__":
    main()