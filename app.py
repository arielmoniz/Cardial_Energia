import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List, Dict
import numpy as np
from dataclasses import dataclass
import sys
import streamlit.runtime.scriptrunner.script_runner as script_runner


# =============== Classes do Sistema Base ===============

@dataclass
class EnergySource:
    name: str
    type: str  # 'grid', 'solar', 'generator'
    capacity: float
    current_output: float
    carbon_footprint: float
    cost_per_kwh: float
    is_active: bool = True

@dataclass
class Equipment:
    id: str
    name: str
    energy_consumption: float
    status: str
    last_maintenance: datetime
    critical: bool

class SensorData:
    def __init__(self):
        self.timestamp = datetime.now()
        self.equipment_readings: Dict[str, float] = {}
        self.environmental_data: Dict[str, float] = {}
        
    def add_equipment_reading(self, equipment_id: str, consumption: float):
        self.equipment_readings[equipment_id] = consumption
        
    def add_environmental_data(self, metric: str, value: float):
        self.environmental_data[metric] = value

class EnergyManagementSystem:
    def __init__(self):
        self.energy_sources: List[EnergySource] = []
        self.equipment: List[Equipment] = []
        self.historical_data: List[SensorData] = []
        
    def add_energy_source(self, source: EnergySource):
        self.energy_sources.append(source)
        
    def add_equipment(self, equip: Equipment):
        self.equipment.append(equip)
        
    def collect_sensor_data(self) -> SensorData:
        """Simula a coleta de dados dos sensores IoT"""
        sensor_data = SensorData()
        
        # Simula leituras de equipamentos
        for equip in self.equipment:
            # Adiciona alguma variação aleatória ao consumo base
            current_consumption = equip.energy_consumption * (1 + random.uniform(-0.1, 0.1))
            sensor_data.add_equipment_reading(equip.id, current_consumption)
            
        # Simula dados ambientais
        sensor_data.add_environmental_data('temperature', random.uniform(20, 35))
        sensor_data.add_environmental_data('humidity', random.uniform(60, 90))
        
        self.historical_data.append(sensor_data)
        return sensor_data
    
    def optimize_energy_distribution(self, sensor_data: SensorData) -> Dict[str, List[EnergySource]]:
        """Otimiza a distribuição de energia entre as fontes disponíveis"""
        total_consumption = sum(sensor_data.equipment_readings.values())
        available_sources = [s for s in self.energy_sources if s.is_active]
        
        # Ordena fontes por pegada de carbono e custo
        available_sources.sort(key=lambda x: (x.carbon_footprint, x.cost_per_kwh))
        
        distribution = {}
        remaining_consumption = total_consumption
        
        for equipment_id, consumption in sensor_data.equipment_readings.items():
            assigned_sources = []
            equipment_consumption = consumption
            
            for source in available_sources:
                if equipment_consumption <= 0:
                    break
                    
                power_from_source = min(source.capacity, equipment_consumption)
                if power_from_source > 0:
                    assigned_sources.append(source)
                    equipment_consumption -= power_from_source
                    
            distribution[equipment_id] = assigned_sources
            
        return distribution
    
    def predict_consumption(self, hours_ahead: int = 24) -> List[float]:
        """Prevê o consumo futuro baseado em dados históricos"""
        if len(self.historical_data) < 24:
            return [0] * hours_ahead
            
        # Simplificação: usa média móvel para previsão
        recent_consumption = [sum(data.equipment_readings.values()) 
                            for data in self.historical_data[-24:]]
        
        prediction = []
        for i in range(hours_ahead):
            # Adiciona tendência e sazonalidade básicas
            base_prediction = np.mean(recent_consumption)
            time_of_day_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 24)
            prediction.append(base_prediction * time_of_day_factor)
            
        return prediction
    
    def calculate_carbon_footprint(self) -> float:
        """Calcula a pegada de carbono total das operações"""
        total_footprint = 0
        
        for data in self.historical_data:
            for equipment_id, consumption in data.equipment_readings.items():
                # Simplificação: usa média ponderada das fontes ativas
                active_sources = [s for s in self.energy_sources if s.is_active]
                if not active_sources:
                    continue
                    
                avg_footprint = sum(s.carbon_footprint for s in active_sources) / len(active_sources)
                total_footprint += consumption * avg_footprint
                
        return total_footprint
    
    def generate_report(self) -> Dict:
        """Gera relatório com métricas principais"""
        if not self.historical_data:
            return {}
            
        latest_data = self.historical_data[-1]
        total_current_consumption = sum(latest_data.equipment_readings.values())
        
        return {
            'timestamp': datetime.now(),
            'total_consumption': total_current_consumption,
            'carbon_footprint': self.calculate_carbon_footprint(),
            'active_sources': [s.name for s in self.energy_sources if s.is_active],
            'equipment_status': {e.id: e.status for e in self.equipment},
            'predicted_consumption_next_24h': self.predict_consumption(),
            'environmental_metrics': latest_data.environmental_data
        }

# =============== Funções do Dashboard ===============

def create_mock_historical_data(ems, hours=24):
    """Cria dados históricos simulados para demonstração"""
    historical_data = []
    start_time = datetime.now() - timedelta(hours=hours)
    
    for i in range(hours):
        current_time = start_time + timedelta(hours=i)
        sensor_data = ems.collect_sensor_data()
        sensor_data.timestamp = current_time
        historical_data.append(sensor_data)
    
    return historical_data

def create_consumption_chart(ems):
    """Cria gráfico de consumo de energia"""
    if not ems.historical_data:
        return None
        
    data = []
    times = []
    for sensor_data in ems.historical_data:
        total_consumption = sum(sensor_data.equipment_readings.values())
        data.append(total_consumption)
        times.append(sensor_data.timestamp)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=data, mode='lines+markers', name='Consumo Total'))
    
    fig.update_layout(
        title='Consumo de Energia ao Longo do Tempo',
        xaxis_title='Hora',
        yaxis_title='Consumo (kWh)',
        height=400
    )
    
    return fig

def create_source_distribution_chart(ems):
    """Cria gráfico de distribuição das fontes de energia"""
    sources = ems.energy_sources
    
    labels = [source.name for source in sources]
    capacities = [source.capacity for source in sources]
    carbon_footprints = [source.carbon_footprint for source in sources]
    
    fig = go.Figure(data=[
        go.Bar(name='Capacidade (kWh)', x=labels, y=capacities),
        go.Bar(name='Pegada de Carbono', x=labels, y=carbon_footprints)
    ])
    
    fig.update_layout(
        title='Distribuição das Fontes de Energia',
        barmode='group',
        height=400
    )
    
    return fig

def create_equipment_status_chart(ems):
    """Cria gráfico de status dos equipamentos"""
    equipment_names = [eq.name for eq in ems.equipment]
    energy_consumption = [eq.energy_consumption for eq in ems.equipment]
    
    fig = px.bar(
        x=equipment_names,
        y=energy_consumption,
        title='Consumo de Energia por Equipamento',
        labels={'x': 'Equipamento', 'y': 'Consumo (kWh)'}
    )
    
    return fig

# =============== Interface Principal ===============

def initialize_session_state():
    """Inicializa o estado da sessão de forma segura"""
    if 'initialized' not in st.session_state:
        ems = EnergyManagementSystem()
        
        # Adiciona fontes de energia
        ems.add_energy_source(EnergySource("Rede Elétrica", "grid", 1000, 800, 0.5, 0.15))
        ems.add_energy_source(EnergySource("Painéis Solares", "solar", 200, 150, 0.1, 0.05))
        ems.add_energy_source(EnergySource("Gerador Diesel", "generator", 500, 0, 0.8, 0.25))
        
        # Adiciona equipamentos
        ems.add_equipment(Equipment("CRANE1", "Guindaste 1", 100, "Operacional", datetime.now(), True))
        ems.add_equipment(Equipment("CRANE2", "Guindaste 2", 120, "Manutenção", datetime.now(), True))
        ems.add_equipment(Equipment("LIGHT1", "Iluminação Terminal 1", 50, "Operacional", datetime.now(), False))
        ems.add_equipment(Equipment("LIGHT2", "Iluminação Terminal 2", 45, "Operacional", datetime.now(), False))
        
        # Cria dados históricos simulados
        ems.historical_data = create_mock_historical_data(ems)
        
        # Armazena no estado da sessão
        st.session_state.ems = ems
        st.session_state.initialized = True

def main():
    try:
        st.set_page_config(
            page_title="Sistema de Gestão Energética - Porto",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        pass  # Ignora se a página já foi configurada

    # Título do dashboard
    st.title("Dashboard de Gestão Energética - Porto")
    
    # Inicializa o estado da sessão
    initialize_session_state()
    
    # Sidebar para controles
    st.sidebar.title("Controles")
    
    # Botão para atualizar dados
    if st.sidebar.button('Atualizar Dados'):
        st.session_state.ems.collect_sensor_data()

    # Controles de fonte de energia
    st.sidebar.subheader("Fontes de Energia")
    for i, source in enumerate(st.session_state.ems.energy_sources):
        source.is_active = st.sidebar.checkbox(
            f"Ativar {source.name}",
            value=source.is_active,
            key=f"source_{i}"
        )

    # Gera relatório atual
    report = st.session_state.ems.generate_report()
    
    # Layout em colunas para métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Consumo Total Atual",
            value=f"{report['total_consumption']:.1f} kWh"
        )
    
    with col2:
        st.metric(
            label="Pegada de Carbono",
            value=f"{report['carbon_footprint']:.1f} kg CO2e"
        )
    
    with col3:
        st.metric(
            label="Fontes Ativas",
            value=len([s for s in st.session_state.ems.energy_sources if s.is_active])
        )
    
    with col4:
        st.metric(
            label="Equipamentos Operacionais",
            value=len([e for e in st.session_state.ems.equipment if e.status == "Operacional"])
        )
    
    # Gráficos
    st.subheader("Análise de Consumo")
    consumption_chart = create_consumption_chart(st.session_state.ems)
    if consumption_chart:
        st.plotly_chart(consumption_chart, use_container_width=True)
    
    # Layout em duas colunas para gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_source_distribution_chart(st.session_state.ems),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_equipment_status_chart(st.session_state.ems),
            use_container_width=True
        )
    
    # Seção de Equipamentos
    st.subheader("Status dos Equipamentos")
    equipment_data = {
        "ID": [e.id for e in st.session_state.ems.equipment],
        "Nome": [e.name for e in st.session_state.ems.equipment],
        "Status": [e.status for e in st.session_state.ems.equipment],
        "Consumo (kWh)": [e.energy_consumption for e in st.session_state.ems.equipment],
        "Crítico": [e.critical for e in st.session_state.ems.equipment],
        "Última Manutenção": [e.last_maintenance for e in st.session_state.ems.equipment]
    }
    df_equipment = pd.DataFrame(equipment_data)
    st.dataframe(df_equipment, use_container_width=True)
    
    # Previsão de Consumo
    st.subheader("Previsão de Consumo (Próximas 24h)")
    prediction = st.session_state.ems.predict_consumption()
    hours = list(range(24))
    fig_prediction = px.line(
        x=hours,
        y=prediction,
        title='Previsão de Consumo',
        labels={'x': 'Horas à Frente', 'y': 'Consumo Previsto (kWh)'}
    )
    st.plotly_chart(fig_prediction, use_container_width=True)

if __name__ == "__main__":
    main()