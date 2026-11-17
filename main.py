
import os, json, time, logging
from datetime import datetime, timezone
import requests
import pandas as pd

BASE_URL = "https://api.bitget.com"
SYMBOL = "BTCUSDT"
PRODUCT_TYPE = "USDT-FUTURES"
STATE_FILE = "bot_state.json"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
MIN_SCORE = int(os.getenv("MIN_SCORE", "7"))
EARLY_SCORE = int(os.getenv("EARLY_SCORE", "4"))
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "1.15"))
FAST_VOLUME_MULT = float(os.getenv("FAST_VOLUME_MULT", "1.30"))
FAST_ADX_MIN = float(os.getenv("FAST_ADX_MIN", "20"))
FAST_MAX_EXTENSION_ATR = float(os.getenv("FAST_MAX_EXTENSION_ATR", "3.0"))
MIN_SMC_SCORE = int(os.getenv("MIN_SMC_SCORE", "1"))
REQUEST_TIMEOUT, CANDLE_LIMIT = 20, 250

FAST_WINDOWS = [
    ("15M", 3, 0.80, 700), ("30M", 6, 1.20, 1000),
    ("1H", 12, 1.60, 1300), ("2H", 24, 2.20, 1800),
    ("3H", 36, 2.80, 2200)
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("BTC-BOT")


def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise RuntimeError("Telegram secrets missing: TELEGRAM_TOKEN / CHAT_ID")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=REQUEST_TIMEOUT)
    log.info("Telegram HTTP status: %s", r.status_code)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram error: {payload}")


def indicators(df):
    df = df.copy()
    c, h, l = df.close, df.high, df.low
    for n in (9, 20, 21, 50, 200):
        df[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["rsi"] = (100 - 100/(1+rs)).fillna(50)
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["vol_avg"] = df.volume.rolling(20).mean().shift(1)
    df["vol_ratio"] = df.volume / df.vol_avg.replace(0, float("nan"))
    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series([u if u > d and u > 0 else 0 for u,d in zip(up,down)], index=df.index)
    minus_dm = pd.Series([d if d > u and d > 0 else 0 for u,d in zip(up,down)], index=df.index)
    atrs = tr.ewm(alpha=1/14, adjust=False).mean()
    pdi = 100*plus_dm.ewm(alpha=1/14, adjust=False).mean()/atrs.replace(0,float("nan"))
    mdi = 100*minus_dm.ewm(alpha=1/14, adjust=False).mean()/atrs.replace(0,float("nan"))
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,float("nan"))
    df["plus_di"], df["minus_di"] = pdi, mdi
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    return df.replace([float("inf"),float("-inf")],float("nan")).dropna().reset_index(drop=True)


def fetch_candles(granularity, limit=CANDLE_LIMIT):
    intervals = {"5m":300000, "15m":900000, "1H":3600000, "4H":14400000}
    if granularity not in intervals:
        raise ValueError(f"Unsupported granularity: {granularity}")
    params = {"symbol":SYMBOL, "productType":PRODUCT_TYPE, "granularity":granularity, "limit":str(limit)}
    error = None
    for attempt in range(3):
        try:
            r = requests.get(BASE_URL+"/api/v2/mix/market/candles", params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if payload.get("code") != "00000":
                raise RuntimeError(payload.get("msg","Bitget API error"))
            rows = [list(x[:7]) for x in payload.get("data",[]) if isinstance(x,(list,tuple)) and len(x)>=7]
            if not rows:
                raise RuntimeError("Empty candle data")
            df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            now = int(time.time()*1000)
            df = df[df.timestamp+intervals[granularity] <= now-3000].reset_index(drop=True)
            if len(df) < 205:
                raise RuntimeError(f"Insufficient closed candles: {len(df)}")
            log.info("%s candles loaded: %s", granularity, len(df))
            return indicators(df)
        except Exception as exc:
            error = exc
            log.warning("%s attempt %s failed: %s", granularity, attempt+1, exc)
            time.sleep(2*(attempt+1))
    raise RuntimeError(f"{granularity} data failed: {error}")


# ---------------- SMC: confirmed swing points, BOS and CHOCH ----------------

def pivots(df, left=3, right=3, limit=15):
    hs, ls = df.high.to_numpy(), df.low.to_numpy()
    sh, sl = [], []
    for i in range(left, len(df)-right):
        hw, lw = hs[i-left:i+right+1], ls[i-left:i+right+1]
        if hs[i] == max(hw) and list(hw).count(hs[i]) == 1:
            sh.append({"index":i,"price":float(hs[i])})
        if ls[i] == min(lw) and list(lw).count(ls[i]) == 1:
            sl.append({"index":i,"price":float(ls[i])})
    return sh[-limit:], sl[-limit:]


def market_structure(df):
    sh, sl = pivots(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    hi = sh[-1]["price"] if sh else float(df.high.iloc[-30:-3].max())
    lo = sl[-1]["price"] if sl else float(df.low.iloc[-30:-3].min())
    trend = "NEUTRAL"
    if len(sh)>=2 and len(sl)>=2:
        hh, hl = sh[-1]["price"]>sh[-2]["price"], sl[-1]["price"]>sl[-2]["price"]
        lh, ll = sh[-1]["price"]<sh[-2]["price"], sl[-1]["price"]<sl[-2]["price"]
        if hh and hl: trend = "BULLISH"
        elif lh and ll: trend = "BEARISH"
    up = bool(last.close>hi and prev.close<=hi)
    down = bool(last.close<lo and prev.close>=lo)
    return {
        "swing_high":hi,"swing_low":lo,"swing_highs":sh,"swing_lows":sl,"trend":trend,
        "bos_up":bool(up and trend!="BEARISH"), "bos_down":bool(down and trend!="BULLISH"),
        "choch_up":bool(up and trend=="BEARISH"), "choch_down":bool(down and trend=="BULLISH")
    }


def equal_levels(points, atr, price, kind):
    if len(points)<2: return []
    tol=max(price*0.001, atr*0.15)
    found=[]
    for i,p in enumerate(points):
        group=[q["price"] for q in points[i:] if abs(q["price"]-p["price"])<=tol]
        if len(group)>=2:
            level=sum(group)/len(group)
            if not any(abs(level-x["level"])<=tol for x in found):
                found.append({"kind":kind,"level":level,"touches":len(group)})
    return found[-3:]


def liquidity_analysis(df, structure):
    last=df.iloc[-1]
    eqh=equal_levels(structure["swing_highs"],float(last.atr),float(last.close),"EQH")
    eql=equal_levels(structure["swing_lows"],float(last.atr),float(last.close),"EQL")
    highs=[x["price"] for x in structure["swing_highs"][-5:]]+[x["level"] for x in eqh]
    lows=[x["price"] for x in structure["swing_lows"][-5:]]+[x["level"] for x in eql]
    sh=next((x for x in sorted(set(highs),reverse=True) if last.high>x and last.close<x),None)
    sl=next((x for x in sorted(set(lows)) if last.low<x and last.close>x),None)
    return {"eqh":eqh,"eql":eql,"high_pools":sorted(set(highs),reverse=True)[:6],
            "low_pools":sorted(set(lows))[:6],"sweep_high":sh is not None,
            "sweep_low":sl is not None,"sweep_high_level":sh,"sweep_low_level":sl}


# ---------------- SMC: Order Block zones and mitigation tracking ----------------

def detect_order_block(df, side, lookback=35):
    start=max(5,len(df)-lookback)
    zones=[]
    for i in range(start+3,len(df)):
        c=df.iloc[i]
        atr=max(float(c.atr),1e-9)
        body=abs(float(c.close-c.open))
        if side=="LONG":
            impulse=c.close>c.open and body>=0.8*atr and c.close>df.high.iloc[i-3:i].max()
        else:
            impulse=c.close<c.open and body>=0.8*atr and c.close<df.low.iloc[i-3:i].min()
        if not impulse: continue
        ob_i=None
        for j in range(i-1,max(start-1,i-8),-1):
            p=df.iloc[j]
            if (side=="LONG" and p.close<p.open) or (side=="SHORT" and p.close>p.open):
                ob_i=j
                break
        if ob_i is None: continue
        ob=df.iloc[ob_i]
        low,high=float(ob.low),float(ob.high)
        later=df.iloc[ob_i+1:]
        touched=bool(((later.low<=high)&(later.high>=low)).any())
        invalid=bool(df.close.iloc[-1]<low) if side=="LONG" else bool(df.close.iloc[-1]>high)
        zones.append({"side":side,"low":low,"high":high,"timestamp":int(ob.timestamp),
                      "impulse_timestamp":int(c.timestamp),"mitigated":touched,"invalidated":invalid})
    if not zones: return None
    active=[z for z in zones if not z["invalidated"]]
    return (active or zones)[-1]


# ---------------- SMC: Fair Value Gaps and mitigation/fill status ----------------

def detect_fvgs(df, lookback=40):
    zones=[]
    for i in range(max(2,len(df)-lookback),len(df)):
        a,b,c=df.iloc[i-2],df.iloc[i-1],df.iloc[i]
        body=abs(float(b.close-b.open))
        if a.high<c.low and b.close>b.open and body>=0.25*max(float(b.atr),1e-9):
            lo,hi=float(a.high),float(c.low)
            later=df.iloc[i+1:]
            touched=bool(((later.low<=hi)&(later.high>=lo)).any())
            filled=bool((later.low<=lo).any())
            zones.append({"side":"BULLISH","low":lo,"high":hi,"timestamp":int(c.timestamp),
                          "mitigated":touched,"filled":filled})
        if a.low>c.high and b.close<b.open and body>=0.25*max(float(b.atr),1e-9):
            lo,hi=float(c.high),float(a.low)
            later=df.iloc[i+1:]
            touched=bool(((later.low<=hi)&(later.high>=lo)).any())
            filled=bool((later.high>=hi).any())
            zones.append({"side":"BEARISH","low":lo,"high":hi,"timestamp":int(c.timestamp),
                          "mitigated":touched,"filled":filled})
    return zones


def latest_fvg(df, side):
    zones=[z for z in detect_fvgs(df) if z["side"]==side]
    active=[z for z in zones if not z["filled"]]
    return (active or zones)[-1] if zones else None


# ---------------- SMC: Premium/Discount dealing range ----------------

def dealing_range(df, lookback=50, price_override=None):
    x=df.tail(lookback)
    hi,lo=float(x.high.max()),float(x.low.min())
    mid=(hi+lo)/2
    span=max(hi-lo,1e-9)
    price=float(df.close.iloc[-1] if price_override is None else price_override)
    return {"high":hi,"low":lo,"midpoint":mid,"premium":price>mid,"discount":price<mid,
            "position_pct":(price-lo)/span*100,
            "ote_discount_low":lo+span*0.62,"ote_discount_high":lo+span*0.79,
            "ote_premium_low":lo+span*0.21,"ote_premium_high":lo+span*0.38}


def smc_engine(d1,d15):
    st1,st15=market_structure(d1),market_structure(d15)
    liq=liquidity_analysis(d15,st15)
    ob_l,ob_s=detect_order_block(d15,"LONG"),detect_order_block(d15,"SHORT")
    fvg_l,fvg_s=latest_fvg(d15,"BULLISH"),latest_fvg(d15,"BEARISH")
    dealing=dealing_range(d1,50,float(d15.close.iloc[-1]))
    price=float(d15.close.iloc[-1]); atr=max(float(d15.atr.iloc[-1]),1e-9)

    def near(zone):
        if not zone or zone.get("invalidated") or zone.get("filled"): return False
        mid=(zone["low"]+zone["high"])/2
        return zone["low"]<=price<=zone["high"] or abs(price-mid)<=atr

    ls,ss,lr,sr=0,0,[],[]
    if st15["bos_up"] or st15["choch_up"]: ls+=2; lr.append("15M bullish BOS/CHOCH")
    if st15["bos_down"] or st15["choch_down"]: ss+=2; sr.append("15M bearish BOS/CHOCH")
    if liq["sweep_low"]: ls+=2; lr.append("Sell-side liquidity sweep")
    if liq["sweep_high"]: ss+=2; sr.append("Buy-side liquidity sweep")
    if near(ob_l): ls+=1; lr.append("Price near bullish OB")
    if near(ob_s): ss+=1; sr.append("Price near bearish OB")
    if fvg_l and not fvg_l["filled"]: ls+=1; lr.append("Bullish FVG active")
    if fvg_s and not fvg_s["filled"]: ss+=1; sr.append("Bearish FVG active")
    if dealing["discount"]: ls+=1; lr.append("Discount dealing range")
    if dealing["premium"]: ss+=1; sr.append("Premium dealing range")
    if liq["eqh"]: sr.append("EQH pool identified")
    if liq["eql"]: lr.append("EQL pool identified")
    return {"structure1":st1,"structure15":st15,"liquidity":liq,"ob_long":ob_l,"ob_short":ob_s,
            "fvg_long":fvg_l,"fvg_short":fvg_s,"dealing":dealing,
            "long_score":ls,"short_score":ss,"long_reasons":lr,"short_reasons":sr,
            "long_ok":ls>=MIN_SMC_SCORE,"short_ok":ss>=MIN_SMC_SCORE}


def timeframe_confirmation(df, lookback=5):
    last=df.iloc[-1]
    ph=float(df.high.iloc[-lookback-1:-1].max())
    pl=float(df.low.iloc[-lookback-1:-1].min())
    rng=max(float(last.high-last.low),1e-9)
    body=abs(float(last.close-last.open))/rng
    long_ok=last.close>ph and last.close>last.ema9>last.ema21 and last.rsi>=52 and last.close>last.open and body>=0.45
    short_ok=last.close<pl and last.close<last.ema9<last.ema21 and last.rsi<=48 and last.close<last.open and body>=0.45
    return {"long":bool(long_ok),"short":bool(short_ok),"price":float(last.close),
            "rsi":float(last.rsi),"timestamp":int(last.timestamp),"ema9":float(last.ema9),
            "ema21":float(last.ema21),"vol_ratio":float(last.vol_ratio),"adx":float(last.adx),
            "plus_di":float(last.plus_di),"minus_di":float(last.minus_di),"body_ratio":body}


def rsi_not_exhausted(side,rsi):
    return 30<rsi<75 if side=="LONG" else 25<rsi<70


def volume_direction_ok(row,side,minimum=MIN_VOLUME_RATIO):
    if float(row.vol_ratio)<minimum: return False
    if side=="LONG": return row.close>row.open and row.plus_di>row.minus_di
    return row.close<row.open and row.minus_di>row.plus_di


# ---------------- Fast Market Monitor ----------------

def fast_momentum(d5,d15,bull4,bear4,bull1,bear1,smc):
    x,y=d5.iloc[-1],d15.iloc[-1]
    found=[]
    for label,n,pct,usd in FAST_WINDOWS:
        if len(d5)<=n: continue
        start=d5.iloc[-n-1]
        move=float(x.close-start.close)
        change=move/float(start.close)*100 if start.close else 0
        if abs(move)>=usd and abs(change)>=pct:
            found.append({"window":label,"move_usd":move,"move_pct":change})
    if not found:
        return {"side":None,"confirmed":False,"window":None,"move_usd":0.0,"move_pct":0.0,
                "volume_ratio":float(x.vol_ratio),"adx":float(x.adx),"rsi":float(x.rsi),
                "atr":float(x.atr),"price":float(x.close),"timestamp":int(x.timestamp),
                "extension_atr":0.0,"reasons":[],"smc":smc}
    strongest=max(found,key=lambda z:abs(z["move_pct"]))
    side="LONG" if strongest["move_usd"]>0 else "SHORT"
    long5=x.close>x.ema9>x.ema21 and x.rsi>=52 and x.plus_di>x.minus_di
    short5=x.close<x.ema9<x.ema21 and x.rsi<=48 and x.minus_di>x.plus_di
    long15=y.close>y.ema9>y.ema21 and y.rsi>=50
    short15=y.close<y.ema9<y.ema21 and y.rsi<=50
    vol=volume_direction_ok(x,side,FAST_VOLUME_MULT)
    adx=float(x.adx)>=FAST_ADX_MIN
    if side=="LONG":
        momentum=long5 and long15
        htf=(bull4 or bull1) and not(bear4 and bear1)
        structure=smc["structure15"]["bos_up"] or smc["structure15"]["choch_up"]
        smc_ok=smc["long_ok"]
    else:
        momentum=short5 and short15
        htf=(bear4 or bear1) and not(bull4 and bull1)
        structure=smc["structure15"]["bos_down"] or smc["structure15"]["choch_down"]
        smc_ok=smc["short_ok"]
    extension=abs(float(x.close-x.ema21)/max(float(x.atr),1e-9))
    not_extended=extension<=FAST_MAX_EXTENSION_ATR
    rsi_ok=rsi_not_exhausted(side,float(x.rsi))
    confirmed=all([vol,adx,momentum,htf,structure,smc_ok,not_extended,rsi_ok])
    reasons=[f"{strongest['window']} impulse","Volume "+("PASS" if vol else "FAIL"),
             "ADX "+("PASS" if adx else "FAIL"),"5M+15M momentum "+("PASS" if momentum else "FAIL"),
             "HTF "+("PASS" if htf else "FAIL"),"BOS/CHOCH "+("PASS" if structure else "FAIL"),
             "SMC score "+("PASS" if smc_ok else "FAIL"),
             "RSI "+("PASS" if rsi_ok else "FAIL"),"Extension "+("PASS" if not_extended else "FAIL")]
    return {"side":side,"confirmed":bool(confirmed),"window":strongest["window"],
            "move_usd":strongest["move_usd"],"move_pct":strongest["move_pct"],
            "volume_ratio":float(x.vol_ratio),"adx":float(x.adx),"rsi":float(x.rsi),
            "atr":float(x.atr),"price":float(x.close),"timestamp":int(x.timestamp),
            "extension_atr":extension,"reasons":reasons,"smc":smc}


def analyze():
    d4,d1,d15,d5=[fetch_candles(t) for t in ("4H","1H","15m","5m")]
    a4,a1,a15,a5=d4.iloc[-1],d1.iloc[-1],d15.iloc[-1],d5.iloc[-1]
    smc=smc_engine(d1,d15)
    s15=smc["structure15"]
    c15,c5=timeframe_confirmation(d15),timeframe_confirmation(d5)
    bull4=bool(a4.ema20>a4.ema50 and a4.close>a4.ema200)
    bear4=bool(a4.ema20<a4.ema50 and a4.close<a4.ema200)
    bull1=bool(a1.ema20>a1.ema50 and a1.close>a1.ema200)
    bear1=bool(a1.ema20<a1.ema50 and a1.close<a1.ema200)

    ls,ss,lr,sr=0,0,[],[]
    def add(ok,points,side,reason):
        nonlocal ls,ss
        if ok:
            if side=="LONG": ls+=points; lr.append(reason)
            else: ss+=points; sr.append(reason)
    add(bull4,2,"LONG","4H bullish"); add(bear4,2,"SHORT","4H bearish")
    add(bull1,2,"LONG","1H bullish"); add(bear1,2,"SHORT","1H bearish")
    add(smc["structure1"]["bos_up"] or smc["structure1"]["choch_up"],2,"LONG","1H BOS/CHOCH")
    add(smc["structure1"]["bos_down"] or smc["structure1"]["choch_down"],2,"SHORT","1H BOS/CHOCH")
    add(s15["bos_up"] or s15["choch_up"],2,"LONG","15M BOS/CHOCH")
    add(s15["bos_down"] or s15["choch_down"],2,"SHORT","15M BOS/CHOCH")
    add(45<=a15.rsi<=68,1,"LONG","15M RSI supportive")
    add(32<=a15.rsi<=55,1,"SHORT","15M RSI supportive")
    if a15.vol_ratio>=MIN_VOLUME_RATIO:
        add(True,1,"LONG" if a15.close>a15.open else "SHORT","15M volume expansion")
    ls+=smc["long_score"]; ss+=smc["short_score"]
    lr += ["SMC: "+x for x in smc["long_reasons"]]
    sr += ["SMC: "+x for x in smc["short_reasons"]]
    if c15["long"]: add(True,1,"LONG","15M confirmation")
    if c5["long"]: add(True,2,"LONG","5M confirmation")
    if c15["short"]: add(True,1,"SHORT","15M confirmation")
    if c5["short"]: add(True,2,"SHORT","5M confirmation")
    early="LONG" if ls>=EARLY_SCORE and ls>ss else ("SHORT" if ss>=EARLY_SCORE and ss>ls else None)
    long_htf=(bull4 or bull1) and not(bear4 and bear1)
    short_htf=(bear4 or bear1) and not(bull4 and bull1)
    long_vol=volume_direction_ok(a15,"LONG") and volume_direction_ok(a5,"LONG")
    short_vol=volume_direction_ok(a15,"SHORT") and volume_direction_ok(a5,"SHORT")
    long_rsi=rsi_not_exhausted("LONG",float(a15.rsi)) and rsi_not_exhausted("LONG",float(a5.rsi))
    short_rsi=rsi_not_exhausted("SHORT",float(a15.rsi)) and rsi_not_exhausted("SHORT",float(a5.rsi))
    final=None
    if c15["long"] and c5["long"] and (s15["bos_up"] or s15["choch_up"]) and long_htf and long_vol and long_rsi and smc["long_ok"] and ls>=MIN_SCORE and ls>ss:
        final="LONG"
    elif c15["short"] and c5["short"] and (s15["bos_down"] or s15["choch_down"]) and short_htf and short_vol and short_rsi and smc["short_ok"] and ss>=MIN_SCORE and ss>ls:
        final="SHORT"
    levels=None
    if final:
        entry=float(a5.close); risk=1.5*float(a5.atr)
        levels=(entry,entry-risk,entry+risk,entry+2*risk,entry+3*risk) if final=="LONG" else (entry,entry+risk,entry-risk,entry-2*risk,entry-3*risk)
    fast=fast_momentum(d5,d15,bull4,bear4,bull1,bear1,smc)
    fast_levels=None
    if fast["confirmed"] and fast["atr"]>0:
        entry=fast["price"]; risk=1.5*fast["atr"]
        fast_levels=(entry,entry-risk,entry+risk,entry+2*risk,entry+3*risk) if fast["side"]=="LONG" else (entry,entry+risk,entry-risk,entry-2*risk,entry-3*risk)
    return {"price":float(a15.close),"early":early,"final":final,"levels":levels,
            "long_score":ls,"short_score":ss,"long_reasons":lr,"short_reasons":sr,
            "bull4":bull4,"bear4":bear4,"bull1":bull1,"bear1":bear1,
            "bos_up":s15["bos_up"],"bos_down":s15["bos_down"],
            "choch_up":s15["choch_up"],"choch_down":s15["choch_down"],
            "confirm15_long":c15["long"],"confirm15_short":c15["short"],
            "confirm5_long":c5["long"],"confirm5_short":c5["short"],
            "rsi":float(a15.rsi),"rsi5":float(a5.rsi),
            "vol15":float(a15.vol_ratio),"vol5":float(a5.vol_ratio),
            "smc":smc,"candle_ts":int(a15.timestamp),"candle5_ts":int(a5.timestamp),
            "fast":fast,"fast_levels":fast_levels}


def fmt(v): return f"{v:,.2f}"
def utc_string(ts): return datetime.fromtimestamp(ts/1000,timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def zone_text(z):
    if not z: return "Not detected"
    status="INVALIDATED" if z.get("invalidated") else ("FILLED" if z.get("filled") else ("MITIGATED/TOUCHED" if z.get("mitigated") else "FRESH"))
    return f"{fmt(z['low'])}-{fmt(z['high'])} [{status}]"

def make_message(r,stage):
    side=r["final"] if stage=="CONFIRMED" else r["early"]
    smc=r["smc"]; liq=smc["liquidity"]; deal=smc["dealing"]
    reasons=r["long_reasons"] if side=="LONG" else r["short_reasons"]
    title="BTC FINAL CONFIRMED SIGNAL" if stage=="CONFIRMED" else "BTC EARLY WARNING - NOT AN ENTRY"
    lines=[title,"","BITGET USDT-M FUTURES",f"BTCUSDT: {side}",f"Price: {fmt(r['price'])}",
           f"LONG Score: {r['long_score']}",f"SHORT Score: {r['short_score']}",
           f"15M RSI: {r['rsi']:.1f}",f"5M RSI: {r['rsi5']:.1f}",
           f"15M Volume: {r['vol15']:.2f}x",f"5M Volume: {r['vol5']:.2f}x","",
           f"4H Bull/Bear: {r['bull4']} / {r['bear4']}",
           f"1H Bull/Bear: {r['bull1']} / {r['bear1']}",
           f"15M BOS UP/DOWN: {r['bos_up']} / {r['bos_down']}",
           f"15M CHOCH UP/DOWN: {r['choch_up']} / {r['choch_down']}",
           f"15M Confirm L/S: {r['confirm15_long']} / {r['confirm15_short']}",
           f"5M Confirm L/S: {r['confirm5_long']} / {r['confirm5_short']}","",
           "----- SMC -----",f"SMC Score L/S: {smc['long_score']} / {smc['short_score']}",
           f"15M Structure: {smc['structure15']['trend']}",
           f"Liquidity Sweep High/Low: {liq['sweep_high']} / {liq['sweep_low']}",
           f"EQH/EQL pools: {len(liq['eqh'])} / {len(liq['eql'])}",
           f"Bullish OB: {zone_text(smc['ob_long'])}",f"Bearish OB: {zone_text(smc['ob_short'])}",
           f"Bullish FVG: {zone_text(smc['fvg_long'])}",f"Bearish FVG: {zone_text(smc['fvg_short'])}",
           f"Dealing Range: {fmt(deal['low'])}-{fmt(deal['high'])}",
           f"Range position: {deal['position_pct']:.1f}% | Premium: {deal['premium']} | Discount: {deal['discount']}"]
    if stage=="CONFIRMED" and r["levels"]:
        e,sl,t1,t2,t3=r["levels"]
        lines += ["","CONFIRMATION: 15M BOS/CHOCH + 15M + 5M + SMC",
                  f"ENTRY: {fmt(e)}",f"STOP LOSS: {fmt(sl)}",f"TP1: {fmt(t1)}",f"TP2: {fmt(t2)}",f"TP3: {fmt(t3)}"]
    else: lines += ["","Wait for final confirmation. Do not enter on Early Warning."]
    lines += ["","Reasons: "+"; ".join(reasons),
              "SMC zones are algorithmic estimates, not institutional order-flow data.",
              "Signals are not guaranteed. Paper-test first.",
              "15M Candle UTC: "+utc_string(r["candle_ts"]),
              "5M Candle UTC: "+utc_string(r["candle5_ts"])]
    return "\n".join(lines)


def make_fast_message(r):
    f=r["fast"]; smc=f["smc"]
    title="BTC FAST MOMENTUM - CONFIRMED ENTRY" if f["confirmed"] else "BTC FAST MOMENTUM WARNING - NOT AN ENTRY"
    lines=[title,"","BITGET USDT-M FUTURES",f"BTCUSDT: {f['side']}",
           f"Current Price: {fmt(f['price'])}",f"Detected Window: {f['window']}",
           f"Move: {f['move_usd']:+,.2f} USD ({f['move_pct']:+.2f}%)",
           f"5M RSI: {f['rsi']:.1f}",f"5M Volume: {f['volume_ratio']:.2f}x",
           f"5M ADX: {f['adx']:.1f}",f"EMA21 extension: {f['extension_atr']:.2f} ATR",
           f"SMC Score L/S: {smc['long_score']} / {smc['short_score']}",
           f"15M Structure: {smc['structure15']['trend']}",
           f"Liquidity sweep H/L: {smc['liquidity']['sweep_high']} / {smc['liquidity']['sweep_low']}",
           f"Bullish OB: {zone_text(smc['ob_long'])}",f"Bearish OB: {zone_text(smc['ob_short'])}",
           f"Bullish FVG: {zone_text(smc['fvg_long'])}",f"Bearish FVG: {zone_text(smc['fvg_short'])}"]
    if f["confirmed"] and r["fast_levels"]:
        e,sl,t1,t2,t3=r["fast_levels"]
        lines += ["",f"ENTRY: {fmt(e)}",f"STOP LOSS: {fmt(sl)}",f"TP1: {fmt(t1)}",f"TP2: {fmt(t2)}",f"TP3: {fmt(t3)}"]
    else: lines += ["","Sharp movement detected, but confirmation incomplete. NOT AN ENTRY. Do not chase."]
    lines += ["","Checks: "+"; ".join(f["reasons"]),"Paper-test first.","5M Candle UTC: "+utc_string(f["timestamp"])]
    return "\n".join(lines)


def diagnostic(r):
    s=r["smc"]; l=s["liquidity"]
    log.info("SCORES LONG=%s SHORT=%s",r["long_score"],r["short_score"])
    log.info("HTF 4H bull/bear=%s/%s 1H bull/bear=%s/%s",r["bull4"],r["bear4"],r["bull1"],r["bear1"])
    log.info("15M BOS U/D=%s/%s CHOCH U/D=%s/%s",r["bos_up"],r["bos_down"],r["choch_up"],r["choch_down"])
    log.info("SMC scores L/S=%s/%s structure=%s",s["long_score"],s["short_score"],s["structure15"]["trend"])
    log.info("SMC EQH=%s EQL=%s sweep H/L=%s/%s",l["eqh"],l["eql"],l["sweep_high"],l["sweep_low"])
    log.info("SMC OB long=%s short=%s",s["ob_long"],s["ob_short"])
    log.info("SMC FVG long=%s short=%s",s["fvg_long"],s["fvg_short"])
    log.info("EARLY=%s FINAL=%s FAST=%s",r["early"],r["final"],r["fast"])


def load_state():
    try:
        with open(STATE_FILE) as f: state=json.load(f)
    except (FileNotFoundError,json.JSONDecodeError): state={}
    for key in ("sent","early_sent","fast_sent"): state.setdefault(key,{})
    return state

def save_state(state):
    with open(STATE_FILE,"w") as f: json.dump(state,f,indent=2)


def main():
    log.info("BTC MULTI-TIMEFRAME + SMC BOT STARTING")
    r=analyze(); diagnostic(r)
    state=load_state()
    if r["early"]:
        key=f"{r['candle_ts']}:EARLY:{r['early']}"
        if key not in state["early_sent"]:
            send_telegram(make_message(r,"EARLY")); state["early_sent"][key]=int(time.time())
    if r["final"] and r["levels"]:
        key=f"{r['candle_ts']}:FINAL:{r['final']}"
        if key not in state["sent"]:
            send_telegram(make_message(r,"CONFIRMED")); state["sent"][key]=int(time.time())
    f=r["fast"]
    if f["side"]:
        stage="FAST_CONFIRMED" if f["confirmed"] else "FAST_WARNING"
        key=f"{f['timestamp']}:{stage}:{f['side']}:{f['window']}"
        if key not in state["fast_sent"]:
            send_telegram(make_fast_message(r)); state["fast_sent"][key]=int(time.time())
    for name in ("sent","early_sent","fast_sent"):
        if len(state[name])>500:
            state[name]=dict(sorted(state[name].items(),key=lambda x:x[1])[-300:])
    save_state(state)
    log.info("BTC BOT FINISHED SUCCESSFULLY")


if __name__=="__main__":
    try: main()
    except Exception:
        log.exception("BTC BOT FAILED")
        raise
