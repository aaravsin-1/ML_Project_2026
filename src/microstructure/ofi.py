"""Order Flow Imbalance (OFI) and Multi-Level OFI (MLOFI) Implementation.

Formalized mathematically by Cont, Kukanov, and Stoikov (2014):
'The Price Impact of Order Book Events'.
Extends to Multi-Level Order Flow Imbalance across M levels with PCA decomposition.
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


class OrderFlowImbalance:
    """Calculates top-level and multi-level Order Flow Imbalance (OFI & MLOFI)."""

    @staticmethod
    def compute_top_level_ofi(
        best_bid_p: np.ndarray,
        best_bid_v: np.ndarray,
        best_ask_p: np.ndarray,
        best_ask_v: np.ndarray
    ) -> np.ndarray:
        """Computes Cont et al. Level-1 Order Flow Imbalance.
        
        Delta_W (Bid side demand):
        - if b(t) > b(t-1): q_b(t)
        - if b(t) == b(t-1): q_b(t) - q_b(t-1)
        - if b(t) < b(t-1): -q_b(t-1)
        
        Delta_V (Ask side supply):
        - if a(t) < a(t-1): q_a(t)
        - if a(t) == a(t-1): q_a(t) - q_a(t-1)
        - if a(t) > a(t-1): -q_a(t-1)
        
        OFI(t) = Delta_W(t) - Delta_V(t)
        """
        n = len(best_bid_p)
        ofi = np.zeros(n, dtype=np.float32)

        # Lagged arrays
        prev_bp = np.roll(best_bid_p, 1)
        prev_bv = np.roll(best_bid_v, 1)
        prev_ap = np.roll(best_ask_p, 1)
        prev_av = np.roll(best_ask_v, 1)

        # First sample has no lag
        prev_bp[0] = best_bid_p[0]
        prev_bv[0] = best_bid_v[0]
        prev_ap[0] = best_ask_p[0]
        prev_av[0] = best_ask_v[0]

        # Bid component Delta_W
        delta_w = np.zeros(n, dtype=np.float32)
        cond_b_up = best_bid_p > prev_bp
        cond_b_eq = best_bid_p == prev_bp
        cond_b_dn = best_bid_p < prev_bp

        delta_w[cond_b_up] = best_bid_v[cond_b_up]
        delta_w[cond_b_eq] = best_bid_v[cond_b_eq] - prev_bv[cond_b_eq]
        delta_w[cond_b_dn] = -prev_bv[cond_b_dn]

        # Ask component Delta_V
        delta_v = np.zeros(n, dtype=np.float32)
        cond_a_dn = best_ask_p < prev_ap  # Ask drops -> aggressive supply or lower ask
        cond_a_eq = best_ask_p == prev_ap
        cond_a_up = best_ask_p > prev_ap  # Ask rises -> supply withdrawn

        delta_v[cond_a_dn] = best_ask_v[cond_a_dn]
        delta_v[cond_a_eq] = best_ask_v[cond_a_eq] - prev_av[cond_a_eq]
        delta_v[cond_a_up] = -prev_av[cond_a_up]

        ofi = delta_w - delta_v
        ofi[0] = 0.0
        return ofi

    @staticmethod
    def compute_multi_level_ofi(
        lob_40: np.ndarray,
        num_levels: int = 10
    ) -> np.ndarray:
        """Computes Multi-Level OFI (MLOFI) across all M depth levels.
        
        Returns:
            mlofi: (N, num_levels) array where column m is OFI at depth level m.
        """
        n_samples = len(lob_40)
        mlofi = np.zeros((n_samples, num_levels), dtype=np.float32)

        for m in range(num_levels):
            idx = m * 4
            ask_p = lob_40[:, idx]
            ask_v = lob_40[:, idx + 1]
            bid_p = lob_40[:, idx + 2]
            bid_v = lob_40[:, idx + 3]
            mlofi[:, m] = OrderFlowImbalance.compute_top_level_ofi(bid_p, bid_v, ask_p, ask_v)

        return mlofi

    @staticmethod
    def reduce_mlofi_pca(
        mlofi: np.ndarray,
        n_components: int = 3,
        pca_model: Optional[PCA] = None
    ) -> Tuple[np.ndarray, PCA]:
        """Applies PCA to multi-level OFI to extract primary order flow factors.
        
        PC1 typically captures aggregate net market flow.
        PC2 captures slope / rotation between top of book and deep book.
        PC3 captures curvature.
        """
        if pca_model is None:
            pca = PCA(n_components=n_components)
            # Standardize or fill zeros
            mlofi_clean = np.nan_to_num(mlofi, nan=0.0, posinf=0.0, neginf=0.0)
            reduced = pca.fit_transform(mlofi_clean)
            return reduced.astype(np.float32), pca
        else:
            mlofi_clean = np.nan_to_num(mlofi, nan=0.0, posinf=0.0, neginf=0.0)
            reduced = pca_model.transform(mlofi_clean)
            return reduced.astype(np.float32), pca_model
