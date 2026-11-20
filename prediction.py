import os
import joblib
import shap
import numpy as np
import pandas as pd


class HousePricePredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "xgb_conformal_house_price_model.pkl")
        feature_path = os.path.join(base_dir, "model_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"❌ 找不到特徵檔：{feature_path}")

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)
        self.conformal_radius, self.alpha = self._extract_conformal_params(self.model)

        # 取得底層 XGBRegressor 物件供 SHAP 使用
        actual_model = self._extract_base_model(self.model)
        self.explainer = shap.TreeExplainer(actual_model)

    def _extract_base_model(self, model_obj):
        actual_model = model_obj
        if isinstance(model_obj, dict):
            for key in ["base_model", "model", "regressor", "estimator"]:
                if key in model_obj and hasattr(model_obj[key], "predict"):
                    actual_model = model_obj[key]
                    break
            if actual_model is model_obj:
                for v in model_obj.values():
                    if hasattr(v, "predict"):
                        actual_model = v
                        break
        if hasattr(actual_model, "model"):
            actual_model = actual_model.model
        return actual_model

    def _extract_conformal_params(self, model_obj):
        """從模型字典中取得 q_conformal 與 alpha（預設 alpha=0.1 代表 90% 信賴區間）"""
        radius = None
        alpha = 0.1
        candidate_keys = [
            "q_conformal", "q_hat_90", "q90", "q_hat", "conformal_q",
            "quantile_90", "radius_90", "margin_90", "margin"
        ]
        if isinstance(model_obj, dict):
            for k in candidate_keys:
                if k in model_obj and isinstance(model_obj[k], (int, float, np.number)):
                    radius = float(model_obj[k])
                    break
            if "alpha" in model_obj and isinstance(model_obj["alpha"], (int, float, np.number)):
                alpha = float(model_obj["alpha"])
        else:
            for attr in candidate_keys:
                if hasattr(model_obj, attr):
                    val = getattr(model_obj, attr)
                    if isinstance(val, (int, float, np.number)):
                        radius = float(val)
                        break
        return radius, alpha

    def _align_features(self, case_dict):
        df = pd.DataFrame([case_dict])
        df = pd.get_dummies(df)

        # 一次性補齊訓練時存在但本次輸入沒有的 One-Hot 欄位
        missing_cols = {col: 0 for col in self.model_features if col not in df.columns}
        if missing_cols:
            missing_df = pd.DataFrame(missing_cols, index=df.index)
            df = pd.concat([df, missing_df], axis=1)

        return df[self.model_features]

    def _feature_to_human(self, feature, value):
        if feature.startswith("district_"):
            return f"位於「{feature.replace('district_', '')}」"
        if feature.startswith("building_type_"):
            return f"建物型態為「{feature.replace('building_type_', '')}」"
        if feature.startswith("main_use_"):
            return f"主要用途為「{feature.replace('main_use_', '')}」"

        human_map = {
            "building_area_sqm": f"建物總面積約 {value:.1f} 平方公尺（約 {value * 0.3025:.1f} 坪）",
            "main_area": f"主建物面積約 {value:.1f} 坪",
            "balcony_area": f"陽台面積約 {value:.1f} 坪",
            "building_age": f"屋齡約 {int(value)} 年",
            "floor": f"位於第 {int(value)} 樓",
            "total_floors": f"建物總樓層 {int(value)} 樓",
            "has_parking": "具備車位" if value == 1 else "未附車位",
            "has_elevator": "設有電梯" if value == 1 else "未設電梯",
        }
        return human_map.get(feature, feature)

    def predict(self, case_dict):
        X = self._align_features(case_dict)

        base_model = self._extract_base_model(self.model)
        pred = float(base_model.predict(X)[0])

        shap_values = self.explainer(X)
        shap_vals = shap_values.values[0]
        feature_names = X.columns
        base_value = float(np.squeeze(self.explainer.expected_value))

        explanation_lines = [
            f"📌 模型基準單價約為 {base_value:.1f} 萬 / 坪，以下條件使價格進行調整："
        ]

        # 依影響幅度排序取前 10 項（略過數值為 0 的未命中類別特徵）
        idx_sorted = np.argsort(np.abs(shap_vals))[::-1]
        shown_count = 0
        for i in idx_sorted:
            feat = feature_names[i]
            val = X.iloc[0][feat]
            shap_val = shap_vals[i]
            if val == 0 and feat.startswith(("district_", "building_type_", "main_use_")):
                continue
            human_text = self._feature_to_human(feat, val)
            direction = "推升" if shap_val > 0 else "下修"
            explanation_lines.append(
                f"👉 {human_text}，使單價約 {direction} {abs(shap_val):.1f} 萬 / 坪"
            )
            shown_count += 1
            if shown_count >= 10:
                break

        explanation_lines.append(
            f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 {pred:.1f} 萬 / 坪。"
        )

        confidence_pct = int(round((1 - self.alpha) * 100))
        lower_bound = None
        upper_bound = None
        if self.conformal_radius is not None:
            lower_bound = max(0.0, pred - self.conformal_radius)
            upper_bound = pred + self.conformal_radius
            explanation_lines.append(
                f"📊 {confidence_pct}% 預測區間（Split Conformal）："
                f"約 {lower_bound:.1f} ～ {upper_bound:.1f} 萬 / 坪（±{self.conformal_radius:.1f} 萬 / 坪）。"
            )
        else:
            explanation_lines.append(f"📊 {confidence_pct}% 預測區間：模型檔中未偵測到區間參數。")

        return {
            "predicted_price": pred,
            "ci_90_lower": lower_bound,
            "ci_90_upper": upper_bound,
            "ci_90_radius": self.conformal_radius,
            "confidence_pct": confidence_pct,
            "explanation": "\n".join(explanation_lines),
        }