import json
import pandas as pd

with open("results/data.json", "r") as f:
    input_data = json.load(f)


λD = input_data["λD"]
λB = input_data["λB"]
rM = input_data["rM"]
rR = input_data["rR"]
hÜ = input_data["hÜ"]
a = input_data["a"]
ζ = input_data["ζ"]
l_Netz = input_data["l_Netz"]
ηPump = input_data["ηPump"]
ρ_water = input_data["ρ_water"]
cp_water = input_data["cp_water"]
ηWüHüs = input_data["ηWüHüs"]
ηWüE = input_data["ηWüE"]
ηVerdichter = input_data["ηVerdichter"]
p_WP_loss = input_data["p_WP_loss"]
T_q_diffmax = input_data["T_q_diffmax"]
ηSpitzenkessel = input_data["ηSpitzenkessel"]
ηBHKW_el = input_data["ηBHKW_el"]
ηBHKW_therm = input_data["ηBHKW_therm"]
T_Wü_delta_r = input_data["T_Wü_delta_r"]
T_Wü_delta_f = input_data["T_Wü_delta_f"]
# ρ_glycol_water = input_data["ρ_glycol_water"]


Tvl_max_vor = input_data["Tvl_max_vor"]
Tvl_min_vor = input_data["Tvl_min_vor"]
Trl_vor = input_data["Trl_vor"]
Tvl_max_nach = input_data["Tvl_max_nach"]
Tvl_min_nach = input_data["Tvl_min_nach"]
Trl_nach = input_data["Trl_nach"]

# Load the CSV file
data = pd.read_csv(
    "Zeitreihen/zeitreihen_22.csv", delimiter=";", dtype={"Einstrahlung_22": str}
)

# Convert the 'Einstrahlung_22' column from strings with commas to floats
data["Einstrahlung_22"] = data["Einstrahlung_22"].str.replace(",", ".").astype(float)

# Get the 'Einstrahlung_22' column
irradiation_data = data["Einstrahlung_22"]

df_input = pd.read_csv("Input_Netz.csv", delimiter=",", decimal=",")
df_input.columns = df_input.columns.str.strip()
df_input["Zeit"] = pd.to_numeric(df_input["Zeit"], errors="coerce")
df_input = df_input.sort_values(by="Zeit")
file_name = "Zeitreihen/zeitreihen_22.csv"
df_zeitreihen = pd.read_csv(file_name, sep=";")


class Erzeuger:
    def __init__(self, color):
        self.color = color

    def calc_output(self, hour, Tvl, Trl):
        pass

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        pass

    def get_class_name(self):
        return self.__class__.__name__

    def calc_co2_emissions(self, power_usage):
        return self.co2_emission_factor * power_usage


class waste_heat(Erzeuger):
    def __init__(
        self,
        Volumenstrom_quelle,
        Abwärmetemperatur,
        color="#639729",
        co2_emission_factor=0,
    ):  # Color for Waermepumpe1
        super().__init__(color)
        self.Volumenstrom_quelle = Volumenstrom_quelle
        self.Abwärmetemperatur = Abwärmetemperatur
        self.co2_emission_factor = co2_emission_factor

    def calc_output(self, hour, Tvl, Trl):
        return (
            ηWüE
            * self.Volumenstrom_quelle
            * ρ_water
            * cp_water
            * (self.Abwärmetemperatur - Trl + T_Wü_delta_r)
        )

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        return 0


class heatpump_1(Erzeuger):
    def __init__(
        self,
        Volumenstrom_quelle,
        T_q,
        Gütegrad,
        Partload,
        color="#1F4E79",
        co2_emission_factor=0.468,
    ):  # Color for Waermepumpe1
        super().__init__(color)
        self.Volumenstrom_quelle = Volumenstrom_quelle
        self.T_q = T_q
        self.Gütegrad = Gütegrad
        self.co2_emission_factor = co2_emission_factor

    def calc_output(self, hour, Tvl, Trl):
        Q_wp_q = (
            self.Volumenstrom_quelle * ρ_water * cp_water * T_q_diffmax
        )  # Wärme aus Quelle
        ε = self.Gütegrad * ((Tvl + 273.15) / (Tvl - self.T_q))
        Q_wp_el = Q_wp_q / (ε - 1)  # Wärme aus Strom
        return Q_wp_q + Q_wp_el

    def calc_COP(self, Tvl, Trl, T_q):
        ε = self.Gütegrad * ((Tvl + 273.15) / (Tvl - self.T_q))
        return ε

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        # Placeholder calculation:
        ε = self.Gütegrad * ((Tvl + 273.15) / (Tvl - self.T_q))
        P = current_last / ε * 1 / (ηVerdichter * p_WP_loss)
        return P


class heatpump_2(Erzeuger):
    def __init__(
        self, Leistung_max, Gütegrad, color="#F7D507", co2_emission_factor=0.468
    ):  # Color for Waermepumpe2
        super().__init__(color)
        self.Leistung_max = Leistung_max
        self.Gütegrad = Gütegrad
        self.co2_emission_factor = co2_emission_factor

    def calc_output(self, hour, Tvl, Trl):
        return self.Leistung_max

    def calc_flowrate(self, Tvl):
        return self.Leistung_max / (ρ_water * cp_water * T_q_diffmax)

    def calc_COP(self, Tvl, Trl, T_q):
        ε = self.Gütegrad * (Tvl + 273.15) / (Tvl - T_q)
        return ε

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        T_q = df_zeitreihen.loc[hour, "Isartemp"]
        ε = self.Gütegrad * (Tvl + 273.15) / (Tvl - T_q)
        P = current_last / ε * 1 / (ηVerdichter * p_WP_loss)
        return P


class geothermal(Erzeuger):
    def __init__(
        self,
        Leistung_max,
        Tgeo,
        h_förder,
        η_geo,
        color="#DD2525",
        co2_emission_factor=0.468,
    ):  # Color for Geothermie
        super().__init__(color)
        self.Leistung_max = Leistung_max
        self.Tgeo = Tgeo
        self.h_förder = h_förder
        self.η_geo = η_geo
        self.co2_emission_factor = co2_emission_factor

    def calc_COP(self, Tvl, Trl, T_q):
        V = 1 / (self.Tgeo - (Trl + T_Wü_delta_r) * ρ_water * cp_water)
        P = V / 3600 * ρ_water * 9.81 * self.h_förder / self.η_geo
        return 1000 / P

    def calc_output(self, hour, Tvl, Trl):
        # Hier Formel einfügen

        return self.Leistung_max

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        # Placeholder calculation:
        V = current_last / (self.Tgeo - (Trl + T_Wü_delta_r) * ρ_water * cp_water)
        P = V / 3600 * ρ_water * 9.81 * self.h_förder / self.η_geo
        return P / 1000


class solarthermal(Erzeuger):
    def __init__(
        self,
        solar_area,
        k_s_1,
        k_s_2,
        α,
        τ,
        color="#F7D507",
        co2_emission_factor=0.468,
    ):  # Color for Solarthermie
        super().__init__(color)
        self.solar_area = solar_area
        self.co2_emission_factor = 0
        self.k_s_1 = k_s_1
        self.k_s_2 = k_s_2
        self.α = α
        self.τ = τ

    def calc_output(self, hour, Tvl, Trl):
        T_u = df_input.loc[hour, "Lufttemp"]
        T_m = (Tvl + T_Wü_delta_f + Trl + T_Wü_delta_r) / 2
        x = (
            (irradiation_data[hour] * self.α * self.τ)
            - (self.k_s_1 * (T_m - T_u))
            - (self.k_s_2 * (T_m - T_u) ** 2)
        ) * self.solar_area
        # T_m = 77
        return [x if x > 0 else 0][0]

    def calc_pressureloss(self, hour, Tvl, Trl):
        return

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        # p_s_delta = ζ_s * 0
        # P_s_el = 0
        return 0


class PLB(Erzeuger):
    def __init__(
        self, Leistung_max, color="#EC9302", co2_emission_factor=0.201
    ):  # Color for Spitzenlastkessel
        super().__init__(color)
        self.Leistung_max = Leistung_max
        self.co2_emission_factor = co2_emission_factor

    def calc_output(self, hour, Tvl, Trl):
        return self.Leistung_max

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        # Placeholder calculation:
        return current_last / ηSpitzenkessel


class CHP(Erzeuger):
    def __init__(
        self, Leistung_max, color="#639729", co2_emission_factor=0.201
    ):  # Color for BHKW
        super().__init__(color)
        self.Leistung_max = Leistung_max
        self.co2_emission_factor = co2_emission_factor

    def calc_output(self, hour, Tvl, Trl):
        return self.Leistung_max

    def calc_electricity_out(self, hour, Tvl, Trl, current_last):
        # Placeholder calculation:
        gas_verbraucht = current_last / ηBHKW_therm
        return 0 - gas_verbraucht * ηBHKW_el

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        # Placeholder calculation:
        gas_verbraucht = current_last / (ηBHKW_therm)
        return gas_verbraucht


class Storage:
    def __init__(
        self,
        max_capacity,
        init_charge,
        charge_efficiency,
        discharge_efficiency,
        max_charge_rate,
        max_discharge_rate,
    ):
        self.max_capacity = max_capacity
        self.current_charge = init_charge
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate

    def charge(self, amount):
        actual_charge = min(
            amount * self.charge_efficiency,
            self.max_capacity - self.current_charge,
            self.max_charge_rate,
        )
        self.current_charge += actual_charge
        return (
            actual_charge / self.charge_efficiency
        )  # return the amount of energy used for charging

    def discharge(self, amount):
        actual_discharge = min(
            amount / self.discharge_efficiency,
            self.current_charge,
            self.max_discharge_rate,
        )
        self.current_charge -= actual_discharge
        return (
            actual_discharge * self.discharge_efficiency
        )  # return the amount of energy produced by discharging
