import datetime

import numpy as np
import pandas as pd
from pyomo.environ import *

from numpy.testing import *

class Calc:
    def load_fixed_params(self) -> None:
        self.n_t: int = 24
        self.init_state_perc: float = 0.0
        self.max_discharge_kw: int = 100
        self.max_charge_kw: int = 100
        self.total_capacity_kwh: int = 200
        self.max_daily_discharge_kwh: int = 200
        self.min_capacity_mwh: int = 0
        self.rte: float = 0.85

    def load_market_data(self, date: str) -> None:
        market_data = pd.read_csv(date + "realtime_zone.csv")

        market_data = market_data[market_data["Name"] == "N.Y.C."]

        market_data["agg_period"] = pd.to_datetime(market_data["Time Stamp"]).dt.ceil(
            "H"
        )

        market_data["beg_time"] = market_data["Time Stamp"].shift(1)
        market_data["end_time"] = market_data["Time Stamp"]

        market_data["beg_time"].iat[0] = (
            market_data["beg_time"].iat[1].split(" ")[0] + " 00:00:00"
        )

        market_data["beg_time"] = pd.to_datetime(market_data["beg_time"])
        market_data["end_time"] = pd.to_datetime(market_data["end_time"])

        self.delta = (market_data["end_time"] - market_data["beg_time"]).astype(
            "timedelta64[m]"
        ).to_numpy() / 60

        self.price = market_data["LBMP ($/MWHr)"].to_numpy() / 1000
        self.n_t = len(self.price)
        self.beg_time = market_data["beg_time"].dt.strftime("%H:%M").to_numpy()
        self.end_time = market_data["end_time"].dt.strftime("%H:%M").to_numpy()

    def opti(self) -> None:
        # create model
        m = ConcreteModel()

        # create variables
        m.charge = Var(
            range(self.n_t), domain=NonNegativeReals, bounds=(0, self.max_charge_kw)
        )

        m.discharge = Var(
            range(self.n_t), domain=NonNegativeReals, bounds=(0, self.max_discharge_kw)
        )

        m.bool = Var(range(self.n_t), domain=Boolean)

        m.SoC = Var(
            range(self.n_t),
            domain=NonNegativeReals,
            bounds=(0, self.total_capacity_kwh),
        )

        # create constraints
        m.cons = ConstraintList()
        m.cons.add(
            m.SoC[0]
            == self.init_state_perc * self.total_capacity_kwh
            + self.rte * self.delta[0] * m.charge[0]
            - self.delta[0] * m.discharge[0]
        )
        
        for t in range(1, self.n_t):
            m.cons.add(
                m.SoC[t]
                == m.SoC[t - 1]
                + self.rte * self.delta[t] * m.charge[t]
                - self.delta[t] * m.discharge[t]
            )

        for t in range(self.n_t):
            m.cons.add(m.charge[t] <= m.bool[t] * self.max_charge_kw)
            m.cons.add(m.discharge[t] <= (1 - m.bool[t]) * self.max_discharge_kw)

        m.cons.add(
            sum(m.discharge[t] * self.delta[t] for t in range(self.n_t))
            <= self.max_daily_discharge_kwh
        )

        revenue = sum(
            ((-m.charge[t] + m.discharge[t]) * self.delta[t] * self.price[t] / 1000)
            for t in range(self.n_t)
        )

        # declare objective
        m.obj = Objective(expr=revenue, sense=maximize)

        results = SolverFactory("glpk").solve(m)

        self.charge = np.array([0.0 + round(m.charge[t](), 2) for t in range(self.n_t)])

        self.discharge = np.array(
            [0.0 + round(m.discharge[t](), 2) for t in range(self.n_t)]
        )

        self.SoC = np.array([0.0 + round(m.SoC[t](), 2) for t in range(self.n_t)])

        self.is_charging = np.array(
            [0.0 + round(m.bool[t](), 0) for t in range(self.n_t)]
        )

    def print_opti_results(self) -> None:
        print("printing opti results(t, discharge, SoC, is_charging):")
        for t in range(self.n_t):
            print(f"{t}, {self.discharge[t]}, {self.SoC[t]}, {self.is_charging[t]}")

    def calc_summary_results(self) -> None:
        self.revenue = round(
            sum(self.delta * (-self.charge + self.discharge) * self.price), 2
        )

        self.total_charging_cost = round(sum(self.delta * self.charge * self.price), 2)

        self.energy_discharged = round(sum(self.delta * self.discharge), 2)

        self.dec = -self.charge + self.discharge
        self.en_traded = np.round(self.dec*self.delta, 2)
        self.price_USD_MWh = self.price * 1000
        self.revenue_hr = np.round(self.delta * self.dec * self.price, 2)

    def results_validation(self, tol=[0.01, 0.05, 0.05]) -> None:

        print('')
        print('Running tests')

        assert_array_less(self.charge, (1+tol[0])*self.max_charge_kw*np.ones(self.n_t))      
        assert_array_less(self.discharge, (1+tol[0])*self.max_discharge_kw*np.ones(self.n_t))

        assert_array_less(self.energy_discharged, (1+tol[0])*self.max_daily_discharge_kwh)

        approx_discharge_price = np.mean(self.price[np.nonzero(self.discharge)])
        approx_charge_price = np.mean(self.price[np.nonzero(self.charge)])
        approx_inc = (approx_discharge_price - approx_charge_price)*self.energy_discharged

        assert_allclose(approx_inc, self.revenue, rtol=tol[1])

        assert_array_less((self.charge > 0) + (self.discharge > 0), 2*np.ones(self.n_t))

        assert_allclose(sum(self.charge*self.delta*self.rte), self.energy_discharged, rtol=tol[2])

        print('tests passed')
        
    def print_summary_results(self) -> None:
        print("")
        print("Printing summary results:")
        print(f"Total revenue generation ($) for the given day: {self.revenue}")
        print(
            f"Total battery storage charging cost ($) for the given day: {self.total_charging_cost}"
        )
        print(
            f"Total battery storage discharged throughput (kWh) for the given day: {self.energy_discharged}"
        )
        print("")
        print("Printing timestep related results:")
        print(f"net start -> finish: \t Pow[kW] \t En[kWh] LBPM[USD/KWh] \t Rev[USD]:")
        for t in range(self.n_t):
            print(
                f"{self.beg_time[t]} -> {self.end_time[t]}: \t {self.dec[t]:6} \t"
                f"{self.en_traded[t]:6} \t {self.price_USD_MWh[t]:6} \t {self.revenue_hr[t]:6}"
            )

    def run(self, date: str) -> None:
        self.load_fixed_params()
        self.load_market_data(date)
        self.opti()
        # self.print_opti_results()
        self.calc_summary_results()
        self.results_validation()
        self.print_summary_results()


Calc().run("20220806")
