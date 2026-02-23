import os
import sys
import warnings

# =============================
# DISABLE INOTIFY & WARNINGS
# =============================
os.environ['STREAMLIT_SERVER_WATCH_FILE_SYSTEM'] = 'false'
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from datetime import timedelta
import copy
from pyswarms.single.global_best import GlobalBestPSO

# =============================
# SET SEED FOR REPRODUCIBILITY
# =============================
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seed(42)

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Stock Price Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 1rem;
    max-width: 1000px;
}
html {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HELPER FUNCTIONS
# =============================
def show_plot(fig, ratio=[1, 5, 1]):
    left, center, right = st.columns(ratio)
    with center:
        st.pyplot(fig, use_container_width=True)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def cleanup_memory():
    """Force cleanup memory"""
    gc.collect()
    K.clear_session()
    tf.keras.backend.clear_session()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("📊 Input Data Saham")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel (Kolom: Date & Close)",
    type=["xlsx", "xls"],
    help="Format: Kolom 'Date' dan 'Close'"
)

section = st.sidebar.radio(
    "📌 Menu",
    ["Informasi Data", "Training & Evaluasi", "Forecast"]
)

# =============================
# VALIDATE & LOAD DATA
# =============================
def load_excel(file):
    try:
        df = pd.read_excel(file)
        df.columns = [c.strip() for c in df.columns]

        if "Date" not in df.columns or "Close" not in df.columns:
            st.error("❌ File harus memiliki kolom: Date dan Close")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df[["Date", "Close"]].dropna().reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ Error membaca file: {str(e)}")
        st.stop()

if uploaded_file is None:
    st.warning("⚠️ Silakan upload file Excel terlebih dahulu.")
    st.info("File harus memiliki 2 kolom: 'Date' dan 'Close'")
    st.stop()

df = load_excel(uploaded_file)

# =============================
# DATA PREPROCESSING
# =============================
@st.cache_data
def preprocess_data(_df):
    """Preprocess data with caching"""
    feature_cols = ["Close"]
    target_col = "Close"
    window = 1
    
    data_features = _df[feature_cols].values
    data_target = _df[[target_col]].values

    values = _df[['Close']].values
    n = len(values)
    n_train = int(n * 0.8)

    scaler_X = MinMaxScaler().fit(data_features[:n_train])
    scaler_y = MinMaxScaler().fit(data_target[:n_train])

    Xs = scaler_X.transform(data_features)
    ys = scaler_y.transform(data_target)

    def make_sequences(X_scaled, y_scaled, window):
        X_seq, y_seq = [], []
        for i in range(window, len(X_scaled)):
            X_seq.append(X_scaled[i-window:i])
            y_seq.append(y_scaled[i])
        return np.array(X_seq), np.array(y_seq)

    X_seq_all, y_seq_all = make_sequences(Xs, ys, window=window)

    train_end_idx = n_train - window
    X_train = X_seq_all[:train_end_idx]
    y_train = y_seq_all[:train_end_idx]
    X_test = X_seq_all[train_end_idx:]
    y_test = y_seq_all[train_end_idx:]

    return X_train, y_train, X_test, y_test, scaler_y, n_train, window

X_train, y_train, X_test, y_test, scaler_y, n_train, window = preprocess_data(df)

# =============================
# BUILD LSTM MODEL
# =============================
def build_lstm_model(input_shape, units, dropout, lr):
    """Build LSTM model dengan memory cleanup"""
    try:
        K.clear_session()
        model = Sequential()
        model.add(LSTM(units=units, input_shape=input_shape, return_sequences=False))
        if dropout is not None and dropout > 0:
            model.add(Dropout(dropout))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss="mse", metrics=['mse'])
        return model
    except Exception as e:
        st.error(f"Error building model: {e}")
        cleanup_memory()
        return None

# =============================
# BASELINE TRAINING
# =============================
def train_baseline():
    """Train Baseline LSTM"""
    try:
        set_seed(42)
        cleanup_memory()
        
        model = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=16,
            dropout=0.5,
            lr=0.001
        )

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            verbose=0,
            shuffle=False
        )

        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = scaler_y.inverse_transform(y_test).flatten()

        baseline_mape = mape(y_true, y_pred)
        
        # CLEANUP
        del model
        cleanup_memory()

        return history, baseline_mape, y_pred, y_true
    
    except Exception as e:
        st.error(f"❌ Error Baseline: {str(e)}")
        cleanup_memory()
        return None, None, None, None

# =============================
# GA TRAINING
# =============================
def train_ga():
    """Train GA-LSTM"""
    try:
        set_seed(42)
        cleanup_memory()
        
        POP_SIZE = 5
        N_GENERATIONS = 5
        MUTATION_RATE = 0.3
        GA_LB = [16, 8, 0.1, 0.0001]
        GA_UB = [160, 256, 0.8, 0.001]

        def init_individual(lb, ub):
            return {
                'units': int(np.random.randint(lb[0], ub[0] + 1)),
                'batch_size': int(np.random.randint(lb[1], ub[1] + 1)),
                'dropout': float(np.random.uniform(lb[2], ub[2])),
                'lr': float(10 ** np.random.uniform(np.log10(lb[3]), np.log10(ub[3])))
            }

        def fitness_ga(indiv, X_tr, y_tr, X_val, y_val, scaler_y):
            try:
                units = int(np.round(indiv['units']))
                batch = int(np.round(indiv['batch_size']))
                dropout = float(indiv['dropout'])
                lr = float(indiv['lr'])
                
                set_seed(42)
                cleanup_memory()
                
                model = build_lstm_model(
                    units=units,
                    dropout=dropout,
                    lr=lr,
                    input_shape=(X_tr.shape[1], X_tr.shape[2])
                )
                model.fit(
                    X_tr, y_tr,
                    epochs=20,
                    batch_size=batch,
                    verbose=0,
                    shuffle=False
                )
                yv_pred = model.predict(X_val, verbose=0)
                yv_pred_orig = scaler_y.inverse_transform(yv_pred).flatten()
                yv_true_orig = scaler_y.inverse_transform(y_val).flatten()
            
                mse_val = mean_squared_error(yv_true_orig, yv_pred_orig)
                
                del model
                cleanup_memory()
                
                return mse_val
                
            except Exception as e:
                cleanup_memory()
                return 1e12

        def crossover(p1, p2, alpha=0.25):
            child = {}
            for k in p1.keys():
                val1 = p1[k]
                val2 = p2[k]
                lower = min(val1, val2) - alpha * abs(val1 - val2)
                upper = max(val1, val2) + alpha * abs(val1 - val2)
                new_val = np.random.uniform(lower, upper)
                          
                if k == 'units':
                    child[k] = int(np.clip(np.round(new_val), GA_LB[0], GA_UB[0]))
                elif k == 'batch_size':
                    child[k] = int(np.clip(np.round(new_val), GA_LB[1], GA_UB[1]))
                elif k == 'dropout':
                    child[k] = float(np.clip(new_val, GA_LB[2], GA_UB[2]))
                elif k == 'lr':
                    child[k] = float(np.clip(new_val, GA_LB[3], GA_UB[3]))
            return child
            
        def mutate(indiv, lb, ub, rate):
            child = copy.deepcopy(indiv)
            if np.random.rand() < rate:
                child['units'] = int(np.random.randint(lb[0], ub[0] + 1))
            if np.random.rand() < rate:
                child['batch_size'] = int(np.random.randint(lb[1], ub[1] + 1))
            if np.random.rand() < rate:
                child['dropout'] = float(np.random.uniform(lb[2], ub[2]))
            if np.random.rand() < rate:
                child['lr'] = float(10 ** np.random.uniform(np.log10(lb[3]), np.log10(ub[3])))
            return child
        
        def build_lstm_model_ga(units, dropout, lr, input_shape):
            cleanup_memory()
            model = Sequential()
            model.add(LSTM(units=units, input_shape=input_shape, return_sequences=False))
            if dropout > 0:
                model.add(Dropout(dropout))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mse'])
            return model
            
        val_frac_for_ga = 0.2
        n_tr_samples_ga = X_train.shape[0]
        n_tr_val_ga = int(n_tr_samples_ga * (1 - val_frac_for_ga))
        
        X_tr_for_ga = X_train[:n_tr_val_ga]
        y_tr_for_ga = y_train[:n_tr_val_ga]
        X_val_for_ga = X_train[n_tr_val_ga:]
        y_val_for_ga = y_train[n_tr_val_ga:]

        population = [init_individual(GA_LB, GA_UB) for _ in range(POP_SIZE)]
        best_mse_ga = np.inf
        best_params_ga = None
        gbest_history_ga = []

        for gen in range(N_GENERATIONS):
            fitness_scores = [
                fitness_ga(ind, X_tr_for_ga, y_tr_for_ga, X_val_for_ga, y_val_for_ga, scaler_y)
                for ind in population
            ]

            order = np.argsort(fitness_scores)
            population = [population[i] for i in order]
            
            if fitness_scores[order[0]] < best_mse_ga:
                best_mse_ga = fitness_scores[order[0]]
                best_params_ga = population[0]
            
            gbest_history_ga.append(best_mse_ga)
            
            elites = population[:3]
            offspring = []
            
            while len(offspring) < POP_SIZE - len(elites):
                idx = np.random.choice(range(len(elites)), 2, replace=False)
                child = crossover(elites[idx[0]], elites[idx[1]])
                child = mutate(child, GA_LB, GA_UB, MUTATION_RATE)
                offspring.append(child)
            
            population = elites + offspring

        best_units_ga = int(np.round(best_params_ga['units']))
        best_lr_ga = float(best_params_ga['lr'])
        best_batch_ga = int(np.round(best_params_ga['batch_size']))
        best_dropout_ga = float(best_params_ga['dropout'])

        set_seed(42)
        cleanup_memory()
        
        final_model_ga = build_lstm_model_ga(
            units=best_units_ga,
            dropout=best_dropout_ga,
            lr=best_lr_ga,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        history_ga = final_model_ga.fit(
            X_train, y_train,
            epochs=100,
            batch_size=best_batch_ga,
            validation_split=0.2,
            verbose=0,
            shuffle=False
        )

        y_pred_scaled_ga = final_model_ga.predict(X_test, verbose=0)
        y_pred_ga = scaler_y.inverse_transform(y_pred_scaled_ga).flatten()
        y_true_ga = scaler_y.inverse_transform(y_test).flatten()

        ga_mape = mape(y_true_ga, y_pred_ga)
        
        del final_model_ga
        cleanup_memory()
        
        return history_ga, ga_mape, y_pred_ga, y_true_ga, gbest_history_ga
    
    except Exception as e:
        st.error(f"❌ Error GA: {str(e)}")
        cleanup_memory()
        return None, None, None, None, None

# =============================
# PSO TRAINING
# =============================
def train_pso():
    """Train PSO-LSTM"""
    try:
        set_seed(42)
        cleanup_memory()

        PSO_N_PARTICLES = 5
        PSO_ITERS = 5
        PSO_OPTIONS = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}

        PSO_BOUNDS = (
            np.array([16, 0.0001, 8, 0.1]),
            np.array([160, 0.001, 256, 0.8])
        )

        val_frac_for_pso = 0.2
        n_tr_samples = X_train.shape[0]
        n_tr_val = int(n_tr_samples * (1 - val_frac_for_pso))

        X_tr_for_pso = X_train[:n_tr_val]
        y_tr_for_pso = y_train[:n_tr_val]
        X_val_for_pso = X_train[n_tr_val:]
        y_val_for_pso = y_train[n_tr_val:]

        def make_pso_obj(X_tr, y_tr, X_va, y_va, scaler_y):
            def obj_fn(particles):
                n_particles = particles.shape[0]
                costs = np.zeros(n_particles)

                for i, p in enumerate(particles):
                    units = int(np.round(p[0]))
                    lr = float(p[1])
                    batch = int(np.round(p[2]))
                    dropout = float(p[3])
                    epochs_fixed = 20

                    try:
                        set_seed(42)
                        cleanup_memory()

                        model = build_lstm_model(
                            input_shape=(X_tr.shape[1], X_tr.shape[2]),
                            units=units,
                            dropout=dropout,
                            lr=lr
                        )

                        model.fit(
                            X_tr, y_tr,
                            epochs=epochs_fixed,
                            batch_size=batch,
                            verbose=0,
                            shuffle=False
                        )

                        yv_pred = model.predict(X_va, verbose=0)
                        yv_pred_orig = scaler_y.inverse_transform(yv_pred).flatten()
                        yv_true_orig = scaler_y.inverse_transform(y_va).flatten()

                        costs[i] = mean_squared_error(yv_true_orig, yv_pred_orig)
                        
                        del model
                        cleanup_memory()

                    except Exception as e:
                        costs[i] = 1e12
                        cleanup_memory()

                return costs
            return obj_fn

        pso_obj = make_pso_obj(
            X_tr_for_pso, y_tr_for_pso,
            X_val_for_pso, y_val_for_pso,
            scaler_y
        )

        optimizer = GlobalBestPSO(
            n_particles=PSO_N_PARTICLES,
            dimensions=4,
            options=PSO_OPTIONS,
            bounds=PSO_BOUNDS
        )

        n_particles, dims = optimizer.swarm.position.shape
        optimizer.swarm.pbest_pos = optimizer.swarm.position.copy()
        optimizer.swarm.pbest_cost = np.full(n_particles, np.inf)

        history_gbest_cost = []
        history_gbest_pos = []

        for it in range(PSO_ITERS):
            costs = pso_obj(optimizer.swarm.position)

            mask = costs < optimizer.swarm.pbest_cost
            optimizer.swarm.pbest_cost[mask] = costs[mask]
            optimizer.swarm.pbest_pos[mask] = optimizer.swarm.position[mask].copy()

            best_idx = np.argmin(optimizer.swarm.pbest_cost)
            optimizer.swarm.best_cost = optimizer.swarm.pbest_cost[best_idx]
            optimizer.swarm.best_pos = optimizer.swarm.pbest_pos[best_idx].copy()

            history_gbest_cost.append(float(optimizer.swarm.best_cost))
            history_gbest_pos.append(optimizer.swarm.best_pos.copy())

            r1 = np.random.rand(*optimizer.swarm.position.shape)
            r2 = np.random.rand(*optimizer.swarm.position.shape)

            optimizer.swarm.velocity = (
                PSO_OPTIONS['w'] * optimizer.swarm.velocity
                + PSO_OPTIONS['c1'] * r1 * (optimizer.swarm.pbest_pos - optimizer.swarm.position)
                + PSO_OPTIONS['c2'] * r2 * (optimizer.swarm.best_pos - optimizer.swarm.position)
            )

            optimizer.swarm.position += optimizer.swarm.velocity
            lb, ub = PSO_BOUNDS
            optimizer.swarm.position = np.clip(optimizer.swarm.position, lb, ub)

        best_pos = history_gbest_pos[-1]
        best_units = int(np.round(best_pos[0]))
        best_lr = float(best_pos[1])
        best_batch = int(np.round(best_pos[2]))
        best_dropout = float(best_pos[3])

        set_seed(42)
        cleanup_memory()

        model_final_pso = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=best_units,
            dropout=best_dropout,
            lr=best_lr
        )

        history_final = model_final_pso.fit(
            X_train, y_train,
            epochs=100,
            batch_size=best_batch,
            validation_split=0.2,
            verbose=0,
            shuffle=False
        )

        y_pred_scaled = model_final_pso.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = scaler_y.inverse_transform(y_test).flatten()

        pso_mape = mape(y_true, y_pred)

        # Simpan model PSO untuk forecast
        st.session_state.model_pso = model_final_pso
        
        return (
            history_final,
            pso_mape,
            y_pred,
            y_true,
            np.array(history_gbest_cost)
        )
    
    except Exception as e:
        st.error(f"❌ Error PSO: {str(e)}")
        cleanup_memory()
        return None, None, None, None, None

# =========================================================
# SESSION STATE
# =========================================================
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model_pso" not in st.session_state:
    st.session_state.model_pso = None

# =========================================================
# BUTTON TRAIN MODEL
# =========================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Training Model")

if st.sidebar.button("▶️ Run Training Model", use_container_width=True):
    with st.spinner("🔄 Training..."):
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # BASELINE
            progress_bar.progress(15)
            status.info("⏳ Training Baseline LSTM...")
            (st.session_state.history_base,
             st.session_state.base_mape,
             st.session_state.y_pred_base,
             st.session_state.y_true_base) = train_baseline()
            status.success("✓ Baseline selesai")
            progress_bar.progress(35)
            
            # GA
            progress_bar.progress(50)
            status.info("⏳ Training GA-LSTM...")
            (st.session_state.history_ga,
             st.session_state.ga_mape,
             st.session_state.y_pred_ga,
             st.session_state.y_true_ga,
             st.session_state.gbest_ga) = train_ga()
            status.success("✓ GA selesai")
            progress_bar.progress(70)
            
            # PSO
            progress_bar.progress(80)
            status.info("⏳ Training PSO-LSTM...")
            (st.session_state.history_pso,
             st.session_state.pso_mape,
             st.session_state.y_pred_pso,
             st.session_state.y_true_pso,
             st.session_state.gbest_pso) = train_pso()
            status.success("✓ PSO selesai")
            progress_bar.progress(100)

            st.session_state.trained = True
            status.empty()
            progress_bar.empty()
            
            st.success("✅ Training semua model selesai!")
            st.balloons()
            
            cleanup_memory()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            cleanup_memory()

# =============================
# SECTION 1 : INFORMASI DATA
# =============================
if section == "Informasi Data":
    st.subheader("📊 Grafik Harga Saham")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], linewidth=2, color='#1f77b4')
    ax.set_title("Pergerakan Harga Saham", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    st.subheader("📈 Statistik Deskriptif")
    col1, col2, col3, col4 = st.columns(4)
    
    stats = df["Close"].describe()
    
    with col1:
        st.metric("Count", f"{int(stats['count'])}")
    with col2:
        st.metric("Mean", f"Rp {stats['mean']:,.0f}")
    with col3:
        st.metric("Min", f"Rp {stats['min']:,.0f}")
    with col4:
        st.metric("Max", f"Rp {stats['max']:,.0f}")
    
    st.dataframe(df.describe().to_frame(), use_container_width=True)

# =============================
# SECTION 2 : TRAINING & EVALUASI
# =============================
elif section == "Training & Evaluasi":
    if not st.session_state.trained:
        st.warning("⚠️ Klik 'Run Training Model' di sidebar untuk memulai training.")
        st.info("Training akan melatih 3 model: Baseline, GA-LSTM, dan PSO-LSTM")
    else:
        history_base = st.session_state.history_base
        history_pso = st.session_state.history_pso
        history_ga = st.session_state.history_ga

        st.subheader("📉 Training vs Validation Loss")
        
        col1, col2, col3 = st.columns(3)

        # BASELINE
        with col1:
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            ax1.plot(history_base.history['loss'], label='Train', linewidth=2)
            ax1.plot(history_base.history['val_loss'], label='Val', linewidth=2)
            ax1.set_title('Baseline LSTM', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
        
        # GA
        with col2:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.plot(history_ga.history['loss'], label='Train', linewidth=2)
            ax2.plot(history_ga.history['val_loss'], label='Val', linewidth=2)
            ax2.set_title('GA-LSTM', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
    
        # PSO
        with col3:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            ax3.plot(history_pso.history['loss'], label='Train', linewidth=2)
            ax3.plot(history_pso.history['val_loss'], label='Val', linewidth=2)
            ax3.set_title('PSO-LSTM', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
        
        # ACTUAL VS PREDICTED
        st.subheader("🎯 Actual vs Predicted Comparison")

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(st.session_state.y_true_base, label="Actual", linewidth=2.5, color='black')
        ax4.plot(st.session_state.y_pred_base, label="Baseline", linewidth=1.5, alpha=0.8)
        ax4.plot(st.session_state.y_pred_pso, label="PSO", linewidth=1.5, alpha=0.8)
        ax4.plot(st.session_state.y_pred_ga, label="GA", linewidth=1.5, alpha=0.8)
        ax4.set_title("Actual vs Predicted (Test Data)", fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Close Price')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        # MAPE TABLE
        st.subheader("📊 MAPE Comparison")

        results = pd.DataFrame({
            "Model": ["Baseline", "PSO", "GA"],
            "MAPE (%)": [
                f"{st.session_state.base_mape:.4f}",
                f"{st.session_state.pso_mape:.4f}",
                f"{st.session_state.ga_mape:.4f}"
            ]
        })

        st.dataframe(results, use_container_width=True)
        
        # Best model info
        mape_values = {
            "Baseline": st.session_state.base_mape,
            "PSO": st.session_state.pso_mape,
            "GA": st.session_state.ga_mape
        }
        best_model = min(mape_values, key=mape_values.get)
        st.success(f"🏆 Model Terbaik: **{best_model}** dengan MAPE = **{mape_values[best_model]:.4f}%**")

# =========================================================
# SECTION 3 : FORECAST
# =========================================================
elif section == "Forecast":
    if not st.session_state.trained:
        st.warning("⚠️ Klik 'Run Training Model' di sidebar terlebih dahulu.")
    else:
        st.subheader("🔮 Forecast Harga Saham (PSO-LSTM)")
        
        future_days = st.slider("📅 Berapa hari ke depan?", 5, 30, 7)

        try:
            last_window = X_test[-1].copy()
            future_preds = []

            model = st.session_state.model_pso
            
            if model is not None:
                for _ in range(future_days):
                    pred = model.predict(last_window.reshape(1, 1, 1), verbose=0)
                    future_preds.append(pred[0, 0])
                    last_window = pred.reshape(1, 1, 1)

                future_preds = scaler_y.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
                
                # BUAT TANGGAL MASA DEPAN
                future_dates = pd.bdate_range(
                    start=df["Date"].iloc[-1],
                    periods=future_days + 1
                )[1:]

                # GRAFIK FORECAST
                fig, ax = plt.subplots(figsize=(12, 5))

                # Data historis
                ax.plot(df["Date"], df["Close"], label="Data Historis", linewidth=2.5, color='#1f77b4')
                
                # Sambungan garis terakhir
                ax.plot(
                    [df["Date"].iloc[-1], future_dates[0]],
                    [df["Close"].iloc[-1], future_preds[0]],
                    linestyle="--",
                    color="orange",
                    linewidth=2
                )
                
                # Forecast
                ax.plot(
                    future_dates,
                    future_preds,
                    label="Forecast",
                    linestyle="--",
                    marker="o",
                    color="red",
                    linewidth=2.5,
                    markersize=6
                )
                
                ax.set_title("Pergerakan Harga Saham + Forecast (PSO-LSTM)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Date")
                ax.set_ylabel("Close Price")
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # TABEL FORECAST
                st.subheader("📋 Tabel Forecast")
                
                future_dates_table = pd.date_range(
                    start=df["Date"].iloc[-1] + timedelta(days=1),
                    periods=future_days
                )

                forecast_df = pd.DataFrame({
                    "Tanggal": future_dates_table.strftime('%Y-%m-%d'),
                    "Forecast (Rp)": [f"{price:,.2f}" for price in future_preds]
                })

                st.dataframe(forecast_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Forecast", f"Rp {future_preds.min():,.0f}")
                with col2:
                    st.metric("Max Forecast", f"Rp {future_preds.max():,.0f}")
                with col3:
                    st.metric("Rata-rata", f"Rp {future_preds.mean():,.0f}")
            else:
                st.error("❌ Model PSO tidak tersedia.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# =============================
# FOOTER
# =============================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Info Aplikasi")
st.sidebar.info(
    "**Stock Price Forecast v2.0**\n\n"
    "✅ Memory Optimized\n"
    "✅ No Inotify Errors\n"
    "✅ 3 Model Comparison\n\n"
    "Made with ❤️"
)
