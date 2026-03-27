
from __future__ import annotations

import json
import os
import hashlib
from enum import Enum
from typing import Any, Dict, Tuple, cast
from datetime import datetime, timezone

from web3 import Web3
from eth_account import Account
from web3.types import TxParams


# =========================
# ---- HARD-CODED CONFIG ---
# =========================
RPC_URL = "https://sepolia.infura.io/v3/ca6d69f9b94e408dbf160e5a7b09fb7e"
PRIVATE_KEY = "ccd7b495c3b16990aeb2cd689b0f31f4fd45947ae87ca3e2dc65e6f1a423f235"  # DEMO: no usar en prod
CONTRACT_ADDRESS = "0xf7b68a576329f104b0855dbbc52d89b85d60f67d"
CHAIN_ID = 11155111  # Sepolia

# ABI real del contrato: addReport(string hash, string summary)
ADD_REPORT_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "hash", "type": "string"},
            {"internalType": "string", "name": "summary", "type": "string"},
        ],
        "name": "addReport",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


# =========================
# Utilities
# =========================
def canon_json_str(obj: Any) -> str:
    """JSON determinista (keys ordenadas, sin espacios extra)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# =========================
# Etapas
# =========================
class Stage(str, Enum):
    DATASET_INGEST = "dataset_ingest"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EXPLAINABILITY_SETUP = "explainability_setup"
    INFERENCE = "inference"


# =========================
# Backend interface + Web3
# =========================
class BaseBackend:
    def write(self, datum: Dict[str, Any]) -> Tuple[bool, str]:
        raise NotImplementedError


class Web3Backend(BaseBackend):
    """
    Envía el JSON 'datum' serializado a contract.addReport(hash, summary_json).
    - hash: sha256(hex) del JSON determinista.
    - summary: el JSON completo del registro.
    """

    def __init__(self, rpc_url: str, contract_address: str, private_key: str, chain_id: int = CHAIN_ID):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            raise RuntimeError("Web3 provider not connected. Revisa el RPC_URL.")

        checksum = Web3.to_checksum_address(contract_address)
        self.contract = self.web3.eth.contract(address=checksum, abi=ADD_REPORT_ABI)

        self.account = Account.from_key(private_key)
        self.chain_id = chain_id

        # Sanity check: que exista bytecode en la dirección
        code = self.web3.eth.get_code(checksum)
        if not code or code in (b"", b"\x00", b"0x"):
            raise RuntimeError(
                f"No hay contrato en la dirección {checksum} en esta red. "
                "Verifica address/red (Sepolia) en el explorador."
            )

    def _eip1559_fees(self) -> Tuple[int, int]:
        """
        Heurística simple para maxFeePerGas y maxPriorityFeePerGas.
        """
        latest_block = self.web3.eth.get_block("latest")
        base_fee = latest_block.get("baseFeePerGas", self.web3.eth.gas_price)
        max_priority = Web3.to_wei(1, "gwei")
        max_fee = base_fee + Web3.to_wei(3, "gwei")
        return max_fee, max_priority

    def write(self, datum: Dict[str, Any]) -> Tuple[bool, str]:
        # 1) Serializar el registro como JSON determinista
        json_str = canon_json_str(datum)

        # 2) Calcular hash SHA-256 del JSON (igual estilo que Android)
        hash_hex = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

        # 3) Armar la transacción EIP-1559
        # IMPORTANTE: usar 'pending' para evitar nonces duplicados
        nonce = self.web3.eth.get_transaction_count(self.account.address, 'pending')
        max_fee, max_priority = self._eip1559_fees()

        tx_params: TxParams = cast(
            TxParams,
            {
                "from": self.account.address,
                "nonce": nonce,
                "chainId": self.chain_id,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": max_priority,
                # "type": 2,  # opcional; web3 normalmente lo infiere
            },
        )

        fn = self.contract.functions.addReport(hash_hex, json_str)
        tx = fn.build_transaction(tx_params)

        # 4) Estimar gas
        gas_estimate = self.web3.eth.estimate_gas({**tx, "from": self.account.address})
        tx["gas"] = int(gas_estimate * 120 // 100)  # +20% buffer

        # 5) Firmar y enviar
        signed = self.account.sign_transaction(tx)

        raw_tx_any: Any = getattr(signed, "rawTransaction", None)
        if raw_tx_any is None:
            raw_tx_any = getattr(signed, "raw_transaction", None)

        if raw_tx_any is None:
            raise RuntimeError("SignedTransaction no tiene rawTransaction ni raw_transaction")

        raw_tx = cast(bytes, raw_tx_any)

        tx_hash = self.web3.eth.send_raw_transaction(raw_tx).hex()
        print(f"[web3] Sent tx: {tx_hash}")
        return True, tx_hash


# =========================
# Logger
# =========================
class ChainLogger:
    def __init__(self, project_id: str, model_version: str, backend: BaseBackend):
        self.project_id = project_id
        self.model_version = model_version
        self.backend = backend
        self.run_id = os.getenv("RUN_ID", "Run_shap_1")

    def register(
        self,
        stage: Stage,
        payload: Dict[str, Any],
        tags: Dict[str, Any] | None = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Registra un evento en blockchain y devuelve (ok, tx_hash, record_enviado).
        """
        record = {
            "project_id": self.project_id,
            "model_version": self.model_version,
            "run_id": "Run_shap_1",
            "stage": stage.value,
            "timestamp_utc": utc_now_iso(),
            "data": payload,
            "tags": tags or {},
            "lib_version": "xai-chain-demo/0.0.1",
        }
        ok, tx_hash = self.backend.write(record)
        return ok, tx_hash, record


# =========================
# Instancia lista para importar + factory
# =========================
def get_chain_logger(
    project_id: str = "xai-bcw-healthcare",
    model_version: str = "v1.0",
    rpc_url: str = RPC_URL,
    contract_address: str = CONTRACT_ADDRESS,
    private_key: str = PRIVATE_KEY,
    chain_id: int = CHAIN_ID,
) -> ChainLogger:
    backend = Web3Backend(rpc_url, contract_address, private_key, chain_id)
    return ChainLogger(project_id, model_version, backend)


# Instancia por defecto al importar (si hay red/contrato válido)
try:
    chain_logger = get_chain_logger()
except Exception as _e:
    # Evita romper import en notebooks si no hay red/contrato; el usuario puede usar get_chain_logger()
    chain_logger = None  # type: ignore


__all__ = [
    "Stage",
    "Web3Backend",
    "ChainLogger",
    "get_chain_logger",
    "chain_logger",
]   