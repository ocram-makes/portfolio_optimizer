import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, Bounds
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

# Configurazione pagina
st.set_page_config(
    page_title="Ottimizzatore Portafoglio Markowitz",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS personalizzato
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #145a8f;
    }
    </style>
    """, unsafe_allow_html=True)


class MarkowitzPortfolio:
    """Ottimizzatore di portafoglio secondo il modello di Markowitz."""
    
    def __init__(self, tickers, start_date='2022-01-01', end_date=None,
                 min_weight=0.01, risk_free_rate=0.02):
        self.tickers = [t.upper() for t in tickers]
        self.n_assets = len(tickers)
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        self.min_weight = min_weight
        self.max_weight = 1.0 - (self.n_assets - 1) * min_weight
        self.risk_free_rate = risk_free_rate
        
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.annual_returns_df = None
        
        self.optimal_weights = None
        self.optimal_return = None
        self.optimal_volatility = None
        self.optimal_sharpe = None
        
        self.frontier_returns = None
        self.frontier_volatilities = None
        self.frontier_sharpes = None

    def _download_from_yahoo(self, ticker):
        """Scarica dati da Yahoo Finance"""
        try:
            etf = yf.Ticker(ticker)
            data = etf.history(period="max", auto_adjust=True)
            if data.empty or len(data) < 10:
                return None
            return data
        except:
            return None

    def _download_from_yahoo_variants(self, ticker):
        """Scarica con varianti ticker europee"""
        variants = [
            ticker, f"{ticker}.L", f"{ticker}.DE", f"{ticker}.MI",
            f"{ticker}.PA", f"{ticker}.AS", f"{ticker}.SW", f"{ticker}.BR",
            f"{ticker}.MC", f"{ticker}.LS"
        ]
        
        for variant in variants:
            try:
                etf = yf.Ticker(variant)
                data = etf.history(period="max", auto_adjust=True)
                if not data.empty and len(data) >= 10:
                    return data
            except:
                continue
        return None

    def download_data(self, progress_bar=None):
        """Scarica dati storici e calcola rendimenti annui reali"""
        all_data_daily = {}
        failed_tickers = []
        
        download_methods = [
            ("Yahoo Finance", self._download_from_yahoo),
            ("Yahoo Finance (varianti)", self._download_from_yahoo_variants),
        ]
        
        for i, ticker in enumerate(self.tickers):
            if progress_bar:
                progress_bar.progress((i + 1) / self.n_assets, 
                                     text=f"Scarico dati per {ticker}...")
            
            data_found = False
            
            for source_name, download_method in download_methods:
                try:
                    data = download_method(ticker)
                    
                    if data is not None and not data.empty and len(data) >= 10:
                        data = data[data.index >= self.start_date]
                        if self.end_date:
                            data = data[data.index <= self.end_date]
                        
                        if len(data) >= 10 and 'Close' in data.columns:
                            all_data_daily[ticker] = data['Close']
                            data_found = True
                            break
                except:
                    continue
            
            if not data_found:
                failed_tickers.append(ticker)
        
        if failed_tickers:
            st.warning(f"⚠️ ETF non trovati: {', '.join(failed_tickers)}")
            self.tickers = [t for t in self.tickers if t not in failed_tickers]
            self.n_assets = len(self.tickers)
        
        if len(all_data_daily) < 2:
            st.error("❌ Servono almeno 2 ETF con dati validi!")
            return False
        
        # Interpolazione e allineamento
        df_daily = pd.DataFrame(all_data_daily)
        df_interpolated = df_daily.fillna(method='ffill', limit=5)
        df_interpolated = df_interpolated.fillna(method='bfill', limit=5)
        df_interpolated = df_interpolated.interpolate(method='linear', limit=10, 
                                                      limit_direction='both')
        df_interpolated = df_interpolated.dropna()
        
        # Calcolo rendimenti annui reali
        df_interpolated['Year'] = df_interpolated.index.year
        annual_returns = {}
        
        for ticker in self.tickers:
            ticker_annual_returns = []
            
            for year in sorted(df_interpolated['Year'].unique()):
                year_data = df_interpolated[df_interpolated['Year'] == year][ticker]
                
                if len(year_data) >= 2:
                    annual_return = (year_data.iloc[-1] / year_data.iloc[0] - 1) * 100
                    ticker_annual_returns.append(annual_return)
            
            annual_returns[ticker] = ticker_annual_returns
        
        self.annual_returns_df = pd.DataFrame(annual_returns)
        self.mean_returns = self.annual_returns_df.mean()
        
        # Matrice di covarianza
        df_weekly = df_interpolated.drop('Year', axis=1).resample('W').last().dropna()
        weekly_returns = df_weekly.pct_change().dropna() * 100
        self.cov_matrix = weekly_returns.cov() * 52
        
        self.data = df_weekly
        self.returns = weekly_returns
        
        return True

    def portfolio_stats(self, weights):
        """Calcola metriche del portafoglio"""
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate * 100) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def negative_sharpe(self, weights):
        """Funzione obiettivo per l'ottimizzazione"""
        return -self.portfolio_stats(weights)[2]

    def optimize_max_sharpe(self):
        """Ottimizza per massimizzare lo Sharpe Ratio"""
        initial_weights = np.array([1.0/self.n_assets] * self.n_assets)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = Bounds(
            lb=[self.min_weight] * self.n_assets,
            ub=[self.max_weight] * self.n_assets
        )
        
        result = minimize(
            fun=self.negative_sharpe,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            st.warning(f"⚠️ Ottimizzazione: {result.message}")
        
        self.optimal_weights = result.x
        self.optimal_return, self.optimal_volatility, self.optimal_sharpe = \
            self.portfolio_stats(self.optimal_weights)
        
        return self.optimal_weights

    def compute_efficient_frontier(self, n_points=100):
        """Calcola la frontiera efficiente"""
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_vols = []
        frontier_rets = []
        frontier_sharpes = []
        
        for target_ret in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.sum(w * self.mean_returns) - target_ret}
            ]
            
            bounds = Bounds(
                lb=[self.min_weight] * self.n_assets,
                ub=[self.max_weight] * self.n_assets
            )
            
            result = minimize(
                fun=lambda w: np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w))),
                x0=np.array([1.0/self.n_assets] * self.n_assets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success:
                ret, vol, sharpe = self.portfolio_stats(result.x)
                frontier_rets.append(ret)
                frontier_vols.append(vol)
                frontier_sharpes.append(sharpe)
        
        self.frontier_returns = np.array(frontier_rets)
        self.frontier_volatilities = np.array(frontier_vols)
        self.frontier_sharpes = np.array(frontier_sharpes)


def create_efficient_frontier_plot(portfolio):
    """Crea grafico della frontiera efficiente"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(portfolio.frontier_volatilities, portfolio.frontier_returns,
                       c=portfolio.frontier_sharpes, cmap='viridis',
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.scatter(portfolio.optimal_volatility, portfolio.optimal_return,
              c='red', s=500, marker='*', edgecolors='black',
              linewidth=2, label='Portafoglio Ottimale', zorder=10)
    
    individual_vols = np.sqrt(np.diag(portfolio.cov_matrix))
    ax.scatter(individual_vols, portfolio.mean_returns.values,
              c='gray', s=100, marker='o', alpha=0.5,
              edgecolors='black', linewidth=1, label='ETF Individuali')
    
    for i, ticker in enumerate(portfolio.tickers):
        ax.annotate(ticker, (individual_vols[i], portfolio.mean_returns.values[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.set_title('Frontiera Efficiente - Modello di Markowitz',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Volatilità Annua (%)', fontsize=13)
    ax.set_ylabel('Rendimento Annuo Atteso (%)', fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    plt.tight_layout()
    
    return fig


def create_allocation_pie(portfolio):
    """Crea grafico a torta dell'allocazione"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sorted_indices = np.argsort(portfolio.optimal_weights)[::-1]
    sorted_tickers = [portfolio.tickers[i] for i in sorted_indices]
    sorted_weights = portfolio.optimal_weights[sorted_indices]
    
    colors = plt.cm.Set3(np.linspace(0, 1, portfolio.n_assets))
    
    wedges, texts, autotexts = ax.pie(
        sorted_weights * 100,
        labels=sorted_tickers,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 11, 'weight': 'bold'},
        pctdistance=0.85
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax.set_title(f'Allocazione Portafoglio Ottimale\nSharpe Ratio: {portfolio.optimal_sharpe:.3f}',
                fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def create_correlation_matrix(portfolio):
    """Crea matrice di correlazione"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = portfolio.returns.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, square=True, linewidths=1,
               cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Matrice di Correlazione\nRendimenti Settimanali',
                fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


# ============================================================================
# INTERFACCIA STREAMLIT
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-title">📊 Ottimizzatore Portafoglio Markowitz</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ottimizzazione di portafoglio ETF con massimizzazione dello Sharpe Ratio</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Configurazione
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        # Lista ETF predefinita
        default_etfs = ['SXLK', 'XDWT', 'CSNDX', 'WTEC', 'NQSE',
                       'XAIX', 'WTAI', 'XNGI', 'CHIP', 'SEME',
                       'SMH', 'XMOV', 'ESIT', 'CTEK', 'HNSC']
        
        st.subheader("📈 Selezione ETF")
        etf_input = st.text_area(
            "Inserisci i ticker degli ETF (uno per riga):",
            value='\n'.join(default_etfs),
            height=300
        )
        
        tickers = [t.strip().upper() for t in etf_input.split('\n') if t.strip()]
        st.info(f"✅ {len(tickers)} ETF selezionati")
        
        st.subheader("📅 Periodo di Analisi")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Data Inizio",
                value=datetime(2022, 1, 1),
                min_value=datetime(2010, 1, 1),
                max_value=datetime.today()
            )
        with col2:
            end_date = st.date_input(
                "Data Fine",
                value=datetime.today(),
                min_value=start_date,
                max_value=datetime.today()
            )
        
        st.subheader("🎯 Parametri Ottimizzazione")
        min_weight = st.slider(
            "Peso minimo per ETF (%)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        ) / 100
        
        risk_free_rate = st.slider(
            "Tasso Risk-Free (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.7,
            step=0.1
        ) / 100
        
        n_frontier_points = st.slider(
            "Punti frontiera efficiente",
            min_value=50,
            max_value=200,
            value=100,
            step=10
        )
        
        st.markdown("---")
        optimize_button = st.button("🚀 AVVIA OTTIMIZZAZIONE", use_container_width=True)
    
    # Main content
    if optimize_button:
        if len(tickers) < 2:
            st.error("❌ Inserisci almeno 2 ETF per l'ottimizzazione!")
            return
        
        # Progress bar
        progress_container = st.empty()
        progress_bar = progress_container.progress(0, text="Inizializzazione...")
        
        try:
            # Creazione portafoglio
            portfolio = MarkowitzPortfolio(
                tickers=tickers,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                min_weight=min_weight,
                risk_free_rate=risk_free_rate
            )
            
            # Download dati
            st.info("📥 Download dati in corso...")
            if not portfolio.download_data(progress_bar):
                st.error("❌ Errore durante il download dei dati")
                return
            
            # Ottimizzazione
            progress_bar.progress(0.7, text="Ottimizzazione in corso...")
            portfolio.optimize_max_sharpe()
            
            # Calcolo frontiera efficiente
            progress_bar.progress(0.9, text="Calcolo frontiera efficiente...")
            portfolio.compute_efficient_frontier(n_points=n_frontier_points)
            
            progress_container.empty()
            st.success("✅ Ottimizzazione completata con successo!")
            
            # ============================================================
            # RISULTATI
            # ============================================================
            
            st.markdown("---")
            st.header("📊 Risultati Ottimizzazione")
            
            # Metriche principali
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Rendimento Annuo",
                    value=f"{portfolio.optimal_return:.2f}%",
                    delta=f"{portfolio.optimal_return - portfolio.mean_returns.mean():.2f}% vs media"
                )
            
            with col2:
                st.metric(
                    label="Volatilità Annua",
                    value=f"{portfolio.optimal_volatility:.2f}%"
                )
            
            with col3:
                st.metric(
                    label="Sharpe Ratio",
                    value=f"{portfolio.optimal_sharpe:.3f}"
                )
            
            with col4:
                st.metric(
                    label="ETF nel Portafoglio",
                    value=f"{portfolio.n_assets}"
                )
            
            # Tabs per visualizzazioni diverse
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Frontiera Efficiente", 
                "🥧 Allocazione", 
                "🔗 Correlazioni",
                "📋 Dettaglio Pesi",
                "📊 Rendimenti Annui"
            ])
            
            with tab1:
                st.subheader("Frontiera Efficiente")
                fig = create_efficient_frontier_plot(portfolio)
                st.pyplot(fig)
                
                st.info("""
                **Interpretazione:**
                - Ogni punto rappresenta un possibile portafoglio
                - Il colore indica lo Sharpe Ratio (più chiaro = migliore)
                - La stella rossa è il portafoglio ottimale (massimo Sharpe)
                - I punti grigi sono i singoli ETF
                """)
            
            with tab2:
                st.subheader("Allocazione Portafoglio Ottimale")
                fig = create_allocation_pie(portfolio)
                st.pyplot(fig)
                
                # Download allocazione
                weights_df = pd.DataFrame({
                    'ETF': portfolio.tickers,
                    'Peso (%)': portfolio.optimal_weights * 100
                }).sort_values('Peso (%)', ascending=False)
                
                csv = weights_df.to_csv(index=False)
                st.download_button(
                    label="📥 Scarica Allocazione (CSV)",
                    data=csv,
                    file_name="allocazione_portafoglio.csv",
                    mime="text/csv"
                )
            
            with tab3:
                st.subheader("Matrice di Correlazione")
                fig = create_correlation_matrix(portfolio)
                st.pyplot(fig)
                
                st.info("""
                **Interpretazione:**
                - Valori vicini a +1: ETF fortemente correlati (si muovono insieme)
                - Valori vicini a -1: ETF inversamente correlati
                - Valori vicini a 0: ETF non correlati (diversificazione ottimale)
                """)
            
            with tab4:
                st.subheader("Dettaglio Allocazione per ETF")
                
                weights_df = pd.DataFrame({
                    'ETF': portfolio.tickers,
                    'Peso (%)': portfolio.optimal_weights * 100,
                    'Rendimento Annuo (%)': portfolio.mean_returns.values,
                    'Volatilità Annua (%)': np.sqrt(np.diag(portfolio.cov_matrix))
                }).sort_values('Peso (%)', ascending=False)
                
                # Formattazione
                styled_df = weights_df.style.format({
                    'Peso (%)': '{:.2f}',
                    'Rendimento Annuo (%)': '{:.2f}',
                    'Volatilità Annua (%)': '{:.2f}'
                }).background_gradient(subset=['Peso (%)'], cmap='Blues')
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Statistiche sommarie
                st.markdown("#### 📊 Statistiche Allocazione")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peso Minimo", f"{weights_df['Peso (%)'].min():.2f}%")
                with col2:
                    st.metric("Peso Massimo", f"{weights_df['Peso (%)'].max():.2f}%")
                with col3:
                    st.metric("Peso Medio", f"{weights_df['Peso (%)'].mean():.2f}%")
            
            with tab5:
                st.subheader("Rendimenti Annui per Anno")
                
                if portfolio.annual_returns_df is not None:
                    # Tabella rendimenti
                    st.dataframe(
                        portfolio.annual_returns_df.style.format("{:.2f}%")
                        .background_gradient(cmap='RdYlGn', axis=None),
                        use_container_width=True
                    )
                    
                    # Grafico rendimenti nel tempo
                    fig, ax = plt.subplots(figsize=(12, 6))
                    portfolio.annual_returns_df.plot(kind='line', ax=ax, marker='o')
                    ax.set_title('Rendimenti Annui per ETF', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Anno')
                    ax.set_ylabel('Rendimento (%)')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Sezione informativa
            st.markdown("---")
            with st.expander("ℹ️ Informazioni sulla Metodologia"):
                st.markdown("""
                ### Modello di Markowitz
                
                Questo ottimizzatore utilizza il modello di Harry Markowitz (Nobel per l'Economia 1990) per:
                
                1. **Massimizzare lo Sharpe Ratio**: rapporto tra rendimento in eccesso e volatilità
                2. **Diversificazione ottimale**: sfrutta le correlazioni tra asset
                3. **Gestione del rischio**: bilancia rendimento atteso e volatilità
                
                ### Calcoli
                
                - **Rendimenti attesi**: Media dei rendimenti annui reali (non annualizzati)
                - **Matrice di covarianza**: Basata su rendimenti settimanali (52 settimane)
                - **Vincoli**: Peso minimo per ogni ETF per garantire diversificazione
                
                ### Fonti Dati
                
                - Yahoo Finance (primaria)
                - Supporto per ticker europei (.L, .DE, .MI, .PA, etc.)
                """)
        
        except Exception as e:
            st.error(f"❌ Errore durante l'ottimizzazione: {str(e)}")
            st.exception(e)
    
    else:
        # Pagina iniziale
        st.markdown("---")
        st.info("""
        ### 👋 Benvenuto nell'Ottimizzatore di Portafoglio!
        
        **Come iniziare:**
        1. Configura i parametri nella barra laterale
        2. Inserisci i ticker degli ETF che vuoi analizzare
        3. Seleziona il periodo di analisi
        4. Clicca su "AVVIA OTTIMIZZAZIONE"
        
        **Caratteristiche:**
        - ✅ Ottimizzazione secondo Markowitz
        - ✅ Massimizzazione Sharpe Ratio
        - ✅ Calcolo rendimenti annui reali
        - ✅ Frontiera efficiente interattiva
        - ✅ Analisi correlazioni
        - ✅ Export risultati CSV
        """)
        
        # Esempio ETF
        st.markdown("### 📋 ETF Predefiniti (Modificabili)")
        col1, col2, col3 = st.columns(3)
        
        default_etfs = ['SXLK', 'XDWT', 'CSNDX', 'WTEC', 'NQSE',
                       'XAIX', 'WTAI', 'XNGI', 'CHIP', 'SEME',
                       'SMH', 'XMOV', 'ESIT', 'CTEK', 'HNSC']
        
        for i, etf in enumerate(default_etfs):
            with [col1, col2, col3][i % 3]:
                st.code(etf, language="text")


if __name__ == "__main__":
    main()
