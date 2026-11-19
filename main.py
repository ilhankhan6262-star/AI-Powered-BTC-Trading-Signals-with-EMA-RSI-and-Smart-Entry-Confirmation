import os, re, json, time, logging, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np

# MASTER GOLD + TOP 10 CRYPTO SMC BOT v4.0
# Bitget USDT-M | 4H/1H/15M/5M | Telegram confirmed signals only
# Alert-only: this bot never places orders.

BASE_URL = 'https://api.bitget.com'
PRODUCT_TYPE = 'USDT-FUTURES'
STATE_FILE = os.getenv('STATE_FILE', 'bot_state.json')
TOKEN = (os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_TOKEN') or '').strip()
CHAT_ID = (os.getenv('TELEGRAM_CHAT_ID') or os.getenv('CHAT_ID') or '').strip()
GOLD = 'XAUUSDT'
TOP_N = 10
MIN_QUOTE_VOLUME = float(os.getenv('MIN_QUOTE_VOLUME', '10000000'))
CANDLE_LIMIT = 240
MAX_WORKERS = max(1, int(os.getenv('MAX_WORKERS', '1')))
API_INTERVAL = float(os.getenv('API_MIN_INTERVAL', '0.60'))
TIMEOUT, RETRIES = 20, 4

MIN_SCORE = 8
MIN_VOLUME_RATIO = 1.10
MIN_ADX = 15.0
MIN_BODY_RATIO = 0.45
MOMENTUM_THRESHOLDS = {'30M': (6, 0.35), '1H': (12, 0.55), '2H': (24, 0.80)}
PIVOT_LEFT, PIVOT_RIGHT = 3, 3
STRUCTURE_MAX_AGE = 2
STRUCTURE_ATR_BUFFER = 0.04
STRONG_5M_VOLUME, STRONG_5M_BODY = 1.45, 0.60
MAX_EXTENSION_STANDARD, MAX_EXTENSION_RAPID = 2.8, 3.5

PREFERRED = ['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','LTCUSDT']
EXCLUDED_BASES = {'USDT','USDC','BUSD','FDUSD','TUSD','USDE','DAI','USDD','UST','EUR','GBP','JPY','XAG'}
# Explicit stock, ETF, leveraged and inverse ETF exclusions. TQQQ is blocked.
BLOCKED_BASES = set('NVDA TSLA AAPL MSFT AMZN META GOOG GOOGL SNDK SOXL SOXS SPCX KORU SNXX MU SKHY SKHYNIX OPENAI QQQ MSTR AMD TQQQ SQQQ UPRO SPXU UVXY SVXY LABU LABD NVDL NVDU TSLL TSLS CONL CONI FNGU FNGD BITX ETHU ETHD YINN YANG ARKK IWM SPY DIA VXX'.split())
BLOCKED_SYMBOLS = {x + 'USDT' for x in BLOCKED_BASES}
BLOCKED_TERMS = ('tokenized stock','tokenised stock','stock token','equity token','share token','stock','equity','shares','etf','leveraged','inverse etf','exchange traded fund','ultra long','ultra short','commodity','forex')
BLOCKED_SUFFIXES = ('3LUSDT','3SUSDT','5LUSDT','5SUSDT','2LUSDT','2SUSDT','BULLUSDT','BEARUSDT','UPUSDT','DOWNUSDT')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('master-smc-bot')
SESSION = requests.Session()
SESSION.headers.update({'User-Agent':'Master-SMC-Alert-Bot/4.0'})
API_LOCK = threading.Lock()
LAST_API_CALL = 0.0


def api_get(path, params=None):
    global LAST_API_CALL
    last_error = None
    for attempt in range(RETRIES):
        wait = min(2 ** (attempt + 1), 30)
        try:
            with API_LOCK:
                delay = API_INTERVAL - (time.monotonic() - LAST_API_CALL)
                if delay > 0: time.sleep(delay)
                LAST_API_CALL = time.monotonic()
                response = SESSION.get(BASE_URL + path, params=params, timeout=TIMEOUT)
            if response.status_code == 400:
                raise ValueError(f'Bitget HTTP 400: {path} | {response.text[:300]}')
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try: wait = max(wait, float(retry_after))
                    except ValueError: pass
                raise requests.HTTPError('Bitget rate limit 429')
            if response.status_code in (500,502,503,504):
                raise requests.HTTPError(f'Bitget server error {response.status_code}')
            response.raise_for_status()
            payload = response.json()
            if str(payload.get('code')) != '00000':
                code, msg = str(payload.get('code')), str(payload.get('msg'))
                if code.startswith('4'): raise ValueError(f'Bitget API parameter error {code}: {msg}')
                raise RuntimeError(f'Bitget API error {code}: {msg}')
            return payload.get('data', [])
        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
            log.warning('API retry %s/%s | %s | %s | wait %.1fs', attempt+1, RETRIES, path, exc, wait)
            if attempt + 1 < RETRIES: time.sleep(wait)
    raise RuntimeError(f'Bitget request failed: {path}: {last_error}')


def get_contracts_tickers():
    contracts = api_get('/api/v2/mix/market/contracts', {'productType':PRODUCT_TYPE})
    tickers = api_get('/api/v2/mix/market/tickers', {'productType':PRODUCT_TYPE})
    return contracts, tickers


def valid_crypto_contract(c):
    symbol = str(c.get('symbol','')).upper()
    base = str(c.get('baseCoin','')).upper()
    quote = str(c.get('quoteCoin','')).upper()
    status = str(c.get('symbolStatus', c.get('status',''))).lower()
    if not symbol.endswith('USDT') or quote != 'USDT' or not base: return False
    if status and status not in ('normal','online','listed'): return False
    if base in EXCLUDED_BASES or base in BLOCKED_BASES or symbol in BLOCKED_SYMBOLS: return False
    if any(symbol.endswith(x) for x in BLOCKED_SUFFIXES): return False
    fields = ('symbolName','symbolDisplayName','baseCoinName','underlying','underlyingType','category','assetType','symbolDesc','businessType')
    metadata = ' '.join(str(c.get(k,'')) for k in fields).lower()
    if any(term in metadata for term in BLOCKED_TERMS): return False
    if re.search(r'(2X|3X|5X|LONG|SHORT|BULL|BEAR)', base): return False
    ctype = str(c.get('contractType','')).lower()
    if ctype and ctype not in ('perpetual','永续'): return False
    return True


def select_symbols():
    contracts, tickers = get_contracts_tickers()
    cmap = {str(c.get('symbol','')).upper():c for c in contracts if c.get('symbol')}
    tmap = {str(t.get('symbol','')).upper():t for t in tickers if t.get('symbol')}
    gold_ok = bool(GOLD in cmap and GOLD in tmap and str(cmap[GOLD].get('quoteCoin','')).upper() == 'USDT')
    if not gold_ok: log.warning('%s unavailable on Bitget %s', GOLD, PRODUCT_TYPE)
    ranked = []
    excluded = []
    for symbol, c in cmap.items():
        if symbol == GOLD: continue
        base = str(c.get('baseCoin','')).upper()
        if base in BLOCKED_BASES or symbol in BLOCKED_SYMBOLS:
            excluded.append(symbol); continue
        if not valid_crypto_contract(c): continue
        t = tmap.get(symbol)
        if not t: continue
        try:
            volume, price = float(t.get('quoteVolume',0) or 0), float(t.get('lastPr',0) or 0)
        except (TypeError,ValueError): continue
        if price > 0 and volume > 0: ranked.append((symbol,volume))
    vmap = dict(ranked)
    eligible = {s for s,_ in ranked}
    selected = [s for s in PREFERRED if s in eligible and vmap.get(s,0) >= MIN_QUOTE_VOLUME][:TOP_N]
    for s,v in sorted(ranked,key=lambda x:x[1],reverse=True):
        if len(selected) >= TOP_N: break
        if s not in selected and v >= MIN_QUOTE_VOLUME: selected.append(s)
    symbols = ([GOLD] if gold_ok else []) + selected
    log.info('Selected %s symbols: %s',len(symbols),', '.join(symbols))
    log.info('Eligible liquid crypto: %s/%s | quote volume floor=%s',len(selected),TOP_N,MIN_QUOTE_VOLUME)
    log.info('Leveraged/equity filter active; TQQQUSDT explicitly blocked')
    if excluded: log.info('Explicitly excluded stock/ETF contracts: %s',', '.join(sorted(set(excluded))))
    return symbols


def add_indicators(df):
    df = df.copy()
    for p in (9,21,50,200): df[f'ema{p}'] = df.close.ewm(span=p,adjust=False).mean()
    delta = df.close.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14,adjust=False).mean(); al = loss.ewm(alpha=1/14,adjust=False).mean()
    rs = ag / al.replace(0,np.nan)
    df['rsi'] = (100 - 100/(1+rs)).fillna(50)
    pc = df.close.shift(1)
    tr = pd.concat([df.high-df.low,(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14,adjust=False).mean()
    up, down = df.high.diff(), -df.low.diff()
    plus = up.where((up>down)&(up>0),0.0); minus = down.where((down>up)&(down>0),0.0)
    safe_atr = df.atr.replace(0,np.nan)
    pdi = 100*plus.ewm(alpha=1/14,adjust=False).mean()/safe_atr
    mdi = 100*minus.ewm(alpha=1/14,adjust=False).mean()/safe_atr
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    df['plus_di']=pdi.fillna(0); df['minus_di']=mdi.fillna(0)
    df['adx']=dx.ewm(alpha=1/14,adjust=False).mean().fillna(0)
    df['volume_ratio']=(df.volume/df.volume.rolling(20).mean().replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    rng=(df.high-df.low).replace(0,np.nan)
    df['body_ratio']=((df.close-df.open).abs()/rng).fillna(0)
    return df


def fetch_candles(symbol, granularity, limit=CANDLE_LIMIT):
    data = api_get('/api/v2/mix/market/candles',{'symbol':symbol,'productType':PRODUCT_TYPE,'granularity':granularity,'limit':str(limit)})
    if not data: raise RuntimeError(f'No candle data: {symbol} {granularity}')
    rows=[]
    for r in data:
        if len(r)<6: continue
        try: rows.append([int(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5]),float(r[6]) if len(r)>6 else 0.0])
        except (TypeError,ValueError): continue
    df=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume','turnover'])
    if df.empty: raise RuntimeError(f'Invalid candles: {symbol} {granularity}')
    df=df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    interval={'5m':300000,'15m':900000,'1H':3600000,'4H':14400000}[granularity]
    df=df[df.timestamp+interval <= int(time.time()*1000)].copy()
    if len(df)<205: raise RuntimeError(f'Not enough closed candles for {symbol} {granularity}: {len(df)}')
    return add_indicators(df.reset_index(drop=True))


def ema_direction(df):
    x=df.iloc[-1]
    if x.close>x.ema50>x.ema200 and x.ema9>x.ema21: return 'BULLISH'
    if x.close<x.ema50<x.ema200 and x.ema9<x.ema21: return 'BEARISH'
    return 'MIXED'


def confirmed_pivots(df):
    highs=[]; lows=[]
    # A pivot is usable only after PIVOT_RIGHT candles have closed.
    for i in range(PIVOT_LEFT,len(df)-PIVOT_RIGHT):
        left=df.iloc[i-PIVOT_LEFT:i]; right=df.iloc[i+1:i+PIVOT_RIGHT+1]
        h=float(df.iloc[i].high); l=float(df.iloc[i].low)
        if h>float(left.high.max()) and h>=float(right.high.max()): highs.append((i,h))
        if l<float(left.low.min()) and l<=float(right.low.min()): lows.append((i,l))
    return highs,lows


def structure_bias(highs,lows):
    if len(highs)<2 or len(lows)<2: return 'UNKNOWN'
    hh=highs[-1][1]>highs[-2][1]; hl=lows[-1][1]>lows[-2][1]
    lh=highs[-1][1]<highs[-2][1]; ll=lows[-1][1]<lows[-2][1]
    if hh and hl: return 'BULLISH'
    if lh and ll: return 'BEARISH'
    return 'RANGE'


def market_structure(df):
    empty={'bos_up':False,'bos_down':False,'choch_up':False,'choch_down':False,'swing_high':0.0,'swing_low':0.0,'bias':'UNKNOWN','event_age':None}
    if len(df)<50: return empty
    # Evaluate recent closed candles independently; each prefix prevents lookahead.
    for age in range(STRUCTURE_MAX_AGE+1):
        end=len(df)-age
        prefix=df.iloc[:end].reset_index(drop=True)
        if len(prefix)<30: continue
        highs,lows=confirmed_pivots(prefix)
        if not highs or not lows: continue
        sh,sl=highs[-1][1],lows[-1][1]
        bias=structure_bias(highs,lows)
        cur=prefix.iloc[-1]; prev=prefix.iloc[-2]
        buf=max(float(cur.atr),1e-12)*STRUCTURE_ATR_BUFFER
        up=float(cur.close)>sh+buf and float(prev.close)<=sh+buf
        dn=float(cur.close)<sl-buf and float(prev.close)>=sl-buf
        bos_up=up and bias!='BEARISH'; choch_up=up and bias=='BEARISH'
        bos_dn=dn and bias!='BULLISH'; choch_dn=dn and bias=='BULLISH'
        # Do not accept a recent break if the latest closed 15M candle
        # has already fallen back through the broken level (failed breakout).
        latest_close = float(df.iloc[-1].close)
        if (bos_up or choch_up) and latest_close <= sh:
            continue
        if (bos_dn or choch_dn) and latest_close >= sl:
            continue
        if bos_up or bos_dn or choch_up or choch_dn:
            return {'bos_up':bool(bos_up),'bos_down':bool(bos_dn),'choch_up':bool(choch_up),'choch_down':bool(choch_dn),'swing_high':float(sh),'swing_low':float(sl),'bias':bias,'event_age':age}
    highs,lows=confirmed_pivots(df)
    if highs and lows: empty.update({'swing_high':float(highs[-1][1]),'swing_low':float(lows[-1][1]),'bias':structure_bias(highs,lows)})
    return empty


def premium_discount(df,lookback=50):
    r=df.iloc[-lookback:]; hi=float(r.high.max()); lo=float(r.low.min()); mid=(hi+lo)/2; price=float(df.iloc[-1].close)
    return {'zone':'DISCOUNT' if price<mid else ('PREMIUM' if price>mid else 'EQUILIBRIUM'),'high':hi,'low':lo,'equilibrium':mid,'price':price}


def liquidity_analysis(df,lookback=25):
    r=df.iloc[-lookback-1:-1]; x=df.iloc[-1]; hi=float(r.high.max()); lo=float(r.low.min())
    return {'sweep_high':bool(x.high>hi and x.close<hi),'sweep_low':bool(x.low<lo and x.close>lo),'equal_high':hi,'equal_low':lo}


def detect_fvg(df):
    if len(df)<4:return None
    a,c=df.iloc[-3],df.iloc[-1]
    if c.low>a.high:return {'type':'BULLISH_FVG','low':float(a.high),'high':float(c.low)}
    if c.high<a.low:return {'type':'BEARISH_FVG','low':float(c.high),'high':float(a.low)}
    return None


def detect_order_block(df,side,lookback=12):
    recent=df.iloc[-(lookback+1):-1]; x=df.iloc[-1]
    if abs(float(x.close-x.open))<0.65*max(float(x.atr),1e-12):return None
    if side=='LONG' and x.close>x.open: opposite=recent[recent.close<recent.open]
    elif side=='SHORT' and x.close<x.open: opposite=recent[recent.close>recent.open]
    else:return None
    if opposite.empty:return None
    c=opposite.iloc[-1]; return {'low':float(c.low),'high':float(c.high)}


def momentum_analysis(df5):
    out={}; price=float(df5.iloc[-1].close)
    for label,(bars,threshold) in MOMENTUM_THRESHOLDS.items():
        start=float(df5.iloc[-bars-1].close); change=(price-start)/start*100 if start>0 else 0.0
        window=df5.iloc[-bars:]; base=df5.iloc[-(bars+21):-bars]
        vavg=float(window.volume.mean()); bavg=float(base.volume.mean()); expansion=vavg/bavg if bavg>0 else 0.0
        direction='UP' if change>0 else ('DOWN' if change<0 else 'NONE')
        atr_pct=float(df5.iloc[-1].atr)/price*100 if price>0 else 0.0
        adaptive=max(threshold,atr_pct*(1.1 if label=='30M' else 1.5))
        out[label]={'direction':direction,'change_pct':change,'volume_expansion':expansion,'strong':bool(abs(change)>=adaptive and expansion>=1.05)}
    return out


def momentum_summary(m):
    return ' | '.join(f"{k}={m[k]['direction']} {m[k]['change_pct']:+.2f}% (Vol x{m[k]['volume_expansion']:.2f}, Strong={m[k]['strong']})" for k in ('30M','1H','2H'))


def candle_confirmation(df,side):
    x,p=df.iloc[-1],df.iloc[-2]; body=abs(float(x.close-x.open)); atr=max(float(x.atr),1e-12)
    br=float(x.body_ratio); vr=float(x.volume_ratio)
    if side=='LONG':
        directional=x.close>x.open
        engulf=directional and p.close<p.open and x.close>=p.open and x.open<=p.close
        breakout=x.close>float(df.high.iloc[-7:-1].max())
        continuation=directional and x.close>p.high
    else:
        directional=x.close<x.open
        engulf=directional and p.close>p.open and x.close<=p.open and x.open>=p.close
        breakout=x.close<float(df.low.iloc[-7:-1].min())
        continuation=directional and x.close<p.low
    return bool(directional and br>=MIN_BODY_RATIO and body>=0.40*atr and (engulf or breakout or continuation) and vr>=MIN_VOLUME_RATIO)


def rapid_retest_confirmation(df,side):
    if len(df)<35:return False,'insufficient 5M history'
    x,p=df.iloc[-1],df.iloc[-2]; prior=df.iloc[-7:-1]; earlier=df.iloc[-13:-7]
    atr=max(float(x.atr),1e-12); br=float(x.body_ratio); vr=float(x.volume_ratio)
    if side=='LONG':
        level=float(earlier.high.max()); prior_break=float(prior.close.max())>level
        retest=prior_break and x.low<=level+0.20*atr and x.close>level and x.close>x.open and br>=0.45 and vr>=1.05
        cont=x.close>float(prior.high.max()) and x.close>x.open and x.close>p.high and br>=0.60 and vr>=1.35
    else:
        level=float(earlier.low.min()); prior_break=float(prior.close.min())<level
        retest=prior_break and x.high>=level-0.20*atr and x.close<level and x.close<x.open and br>=0.45 and vr>=1.05
        cont=x.close<float(prior.low.min()) and x.close<x.open and x.close<p.low and br>=0.60 and vr>=1.35
    if retest:return True,'5M breakout retest'
    if cont:return True,'5M strong continuation'
    return False,'no valid retest/continuation'


def strong_5m(df,side):
    x=df.iloc[-1]; directional=x.close>x.open if side=='LONG' else x.close<x.open
    return bool(directional and x.volume_ratio>=STRONG_5M_VOLUME and x.body_ratio>=STRONG_5M_BODY and abs(float(x.close-x.open))>=0.60*max(float(x.atr),1e-12))


def trade_levels(df,side):
    x=df.iloc[-1]; entry=float(x.close); atr=max(float(x.atr),entry*0.0005); risk=max(1.50*atr,entry*0.0015)
    if side=='LONG':return {'entry':entry,'area_low':entry-0.20*atr,'area_high':entry+0.15*atr,'sl':entry-risk,'tp1':entry+risk,'tp2':entry+2*risk,'tp3':entry+3*risk}
    return {'entry':entry,'area_low':entry-0.15*atr,'area_high':entry+0.20*atr,'sl':entry+risk,'tp1':entry-risk,'tp2':entry-2*risk,'tp3':entry-3*risk}


def analyze_symbol(symbol):
    def reject(reason,diag=''):
        log.info('DIAGNOSTIC %s | REJECT: %s%s',symbol,reason,(' | '+diag) if diag else '')
        return None
    try:
        d4=fetch_candles(symbol,'4H'); d1=fetch_candles(symbol,'1H'); d15=fetch_candles(symbol,'15m'); d5=fetch_candles(symbol,'5m')
        t4,t1=ema_direction(d4),ema_direction(d1); st=market_structure(d15); pdz=premium_discount(d15); liq=liquidity_analysis(d15); mom=momentum_analysis(d5); x=d5.iloc[-1]
        long_struct=st['bos_up'] or st['choch_up']; short_struct=st['bos_down'] or st['choch_down']
        long_mom=any(v['strong'] and v['direction']=='UP' for v in mom.values()); short_mom=any(v['strong'] and v['direction']=='DOWN' for v in mom.values())
        lc=int(t4=='BULLISH')+int(t1=='BULLISH'); sc=int(t4=='BEARISH')+int(t1=='BEARISH')
        diag=f'4H={t4}, 1H={t1}, 15M BOS U/D={st["bos_up"]}/{st["bos_down"]}, CHOCH U/D={st["choch_up"]}/{st["choch_down"]}, bias={st["bias"]}, event_age={st["event_age"]}, PD={pdz["zone"]}, 5M vol={x.volume_ratio:.2f}, body={x.body_ratio:.2f}, ADX={x.adx:.1f}, {momentum_summary(mom)}'
        side=None; rapid=False
        if lc==2 and long_struct:side='LONG'
        elif sc==2 and short_struct:side='SHORT'
        elif lc>=1 and sc==0 and long_struct and long_mom:side='LONG'; rapid=True
        elif sc>=1 and lc==0 and short_struct and short_mom:side='SHORT'; rapid=True
        if side is None:
            why=[]
            if lc==0 and sc==0:why.append('4H/1H trend mixed')
            elif lc<2 and sc<2:why.append('HTF not aligned or opposing')
            if not long_struct and not short_struct:why.append('recent 15M swing BOS/CHOCH not confirmed')
            if long_struct and lc>=1 and not long_mom:why.append('bullish momentum weak')
            if short_struct and sc>=1 and not short_mom:why.append('bearish momentum weak')
            return reject('; '.join(why) or 'directional setup not aligned',diag)
        ema15bull=d15.iloc[-1].ema9>d15.iloc[-1].ema21; ema15bear=d15.iloc[-1].ema9<d15.iloc[-1].ema21
        if side=='LONG' and not ema15bull:return reject('15M EMA disagrees with LONG',diag)
        if side=='SHORT' and not ema15bear:return reject('15M EMA disagrees with SHORT',diag)
        standard=candle_confirmation(d5,side); rapid_ok,rapid_reason=rapid_retest_confirmation(d5,side); strong5=strong_5m(d5,side)
        if rapid:
            if not (standard or rapid_ok):return reject('rapid route: 5M confirmation failed',diag+' | '+rapid_reason)
        elif not standard:return reject('standard route: 5M closed-candle confirmation failed',diag)
        aligned_mom=long_mom if side=='LONG' else short_mom
        if not (aligned_mom or strong5):return reject('momentum gate failed',diag)
        score=2 if (side=='LONG' and lc==2) or (side=='SHORT' and sc==2) else 1
        score+=2 if (side=='LONG' and st['bos_up']) or (side=='SHORT' and st['bos_down']) else 1
        score+=2 if standard or rapid_ok else 0
        score+=int(float(x.adx)>=MIN_ADX)+int(float(x.volume_ratio)>=MIN_VOLUME_RATIO)+int(aligned_mom)
        score+=int((side=='LONG' and liq['sweep_low']) or (side=='SHORT' and liq['sweep_high']))
        score+=int((side=='LONG' and pdz['zone']=='DISCOUNT') or (side=='SHORT' and pdz['zone']=='PREMIUM'))
        score+=1 # 15M EMA alignment passed.
        if score<MIN_SCORE:return reject(f'score too low {score}/12; required {MIN_SCORE}',diag)
        price=float(x.close); atr=max(float(x.atr),1e-12); extension=abs(price-float(x.ema21))/atr
        limit=MAX_EXTENSION_RAPID if rapid else MAX_EXTENSION_STANDARD
        if extension>limit:return reject(f'overextended from EMA21 {extension:.2f} ATR; limit {limit:.2f}',diag)
        result={'symbol':symbol,'side':side,'score':score,**trade_levels(d5,side),'candle_ts':int(x.timestamp),'trend4':t4,'trend1':t1,
          'ema_values':{tf:{f'EMA{n}':float(frame.iloc[-1][f'ema{n}']) for n in (9,21,50,200)} for tf,frame in [('4H',d4),('1H',d1),('15M',d15),('5M',d5)]},
          'pd_zone':pdz['zone'],'momentum':mom,'fvg':detect_fvg(d15),'ob':detect_order_block(d15,side),'liquidity':liq,'structure':st,
          'entry_mode':'RAPID RETEST/CONTINUATION' if rapid else 'STANDARD SMC','confirmation':rapid_reason if rapid and rapid_ok else '5M closed-candle confirmation',
          'rsi5':float(x.rsi),'adx5':float(x.adx),'volume_ratio5':float(x.volume_ratio)}
        log.info('DIAGNOSTIC %s | CONFIRMED %s | score=%s/12 | %s',symbol,side,score,diag)
        return result
    except Exception as exc:
        log.exception('Analysis failed for %s: %s',symbol,exc)
        return None


def load_state():
    try:
        with open(STATE_FILE,'r',encoding='utf-8') as f: state=json.load(f)
        return state if isinstance(state,dict) else {}
    except FileNotFoundError:return {}
    except Exception as exc:log.warning('State read failed: %s',exc); return {}


def save_state(state):
    tmp=STATE_FILE+'.tmp'
    with open(tmp,'w',encoding='utf-8') as f:json.dump(state,f,indent=2)
    os.replace(tmp,STATE_FILE)


def fmt(v):
    v=float(v)
    if v>=1000:return f'{v:,.2f}'
    if v>=1:return f'{v:.5f}'.rstrip('0').rstrip('.')
    return f'{v:.8f}'.rstrip('0').rstrip('.')


def build_message(s):
    direction='🟢 LONG' if s['side']=='LONG' else '🔴 SHORT'
    ema=[]
    for tf in ('4H','1H','15M','5M'):
        e=s['ema_values'][tf]; ema.append(f"EMA {tf}: 9={fmt(e['EMA9'])} | 21={fmt(e['EMA21'])} | 50={fmt(e['EMA50'])} | 200={fmt(e['EMA200'])}")
    mom=[]
    for k in ('30M','1H','2H'):
        m=s['momentum'][k]; mom.append(f"{k}: {m['direction']} {m['change_pct']:+.2f}% | Vol x{m['volume_expansion']:.2f} | Strong={m['strong']}")
    st=s['structure']; fvg=s['fvg']; ob=s['ob']; liq=s['liquidity']
    fvg_txt=f"{fvg['type']}: {fmt(fvg['low'])} - {fmt(fvg['high'])}" if fvg else 'No fresh FVG detected'
    ob_txt=f"{fmt(ob['low'])} - {fmt(ob['high'])}" if ob else 'No valid OB detected'
    liq_txt=('Sell-side sweep detected' if liq['sweep_low'] else 'No sell-side sweep') if s['side']=='LONG' else ('Buy-side sweep detected' if liq['sweep_high'] else 'No buy-side sweep')
    return (f'✅ FINAL CONFIRMED {direction}\n\nCoin: {s["symbol"]}\n\n'
      f'Entry Area: {fmt(s["area_low"])} - {fmt(s["area_high"])}\nEntry: {fmt(s["entry"])}\n\n'
      f'Stop Loss: {fmt(s["sl"])}\nTP1: {fmt(s["tp1"])}\nTP2: {fmt(s["tp2"])}\nTP3: {fmt(s["tp3"])}\n\n'
      f'Signal Score: {s["score"]}/12\nEntry Mode: {s["entry_mode"]}\nConfirmation: {s["confirmation"]}\n\n'
      f'4H Trend: {s["trend4"]}\n1H Trend: {s["trend1"]}\nPremium/Discount: {s["pd_zone"]}\n\n'
      f'15M Structure: BOS UP={st["bos_up"]} | BOS DOWN={st["bos_down"]} | CHOCH UP={st["choch_up"]} | CHOCH DOWN={st["choch_down"]}\n'
      f'Structure Bias: {st["bias"]} | Event age: {st["event_age"]}\nLiquidity: {liq_txt}\nFVG: {fvg_txt}\nOrder Block: {ob_txt}\n\n'
      + '\n'.join(ema) + '\n\nMomentum:\n' + '\n'.join(mom) + '\n\n'
      + f'5M RSI: {s["rsi5"]:.2f}\n5M ADX: {s["adx5"]:.2f}\n5M Volume Ratio: x{s["volume_ratio5"]:.2f}\n\n'
      + 'Risk Note: Algorithmic analysis, not a guarantee of profit. No orders are placed automatically.')


def send_telegram(message):
    if not TOKEN or not CHAT_ID:
        log.error('Telegram credentials missing; message not delivered.')
        log.info('\n%s',message); return False
    try:
        r=SESSION.post(f'https://api.telegram.org/bot{TOKEN}/sendMessage',json={'chat_id':CHAT_ID,'text':message,'disable_web_page_preview':True},timeout=TIMEOUT)
        r.raise_for_status(); payload=r.json()
        if not payload.get('ok'):raise RuntimeError(str(payload))
        return True
    except Exception as exc:log.error('Telegram send failed: %s',exc); return False


def main():
    log.info('Starting Master Gold + Top 10 Crypto SMC Bot v4.0')
    log.info('API pacing=%.2fs | workers=%s | 4H/1H/15M/5M',API_INTERVAL,MAX_WORKERS)
    log.info('Closed-candle analysis + confirmed swing-pivot BOS/CHOCH enabled')
    log.info('5M confirmation + stricter retest/continuation enabled')
    log.info('Telegram: FINAL CONFIRMED SIGNALS ONLY | no early-warning messages')
    log.info('Stock/equity/tokenized/leveraged/inverse ETF filters enabled')
    if not TOKEN or not CHAT_ID:log.warning('Telegram secrets missing; check GitHub Actions repository secrets.')
    try:symbols=select_symbols()
    except Exception as exc:log.exception('Symbol selection failed: %s',exc); return
    if not symbols:log.error('No eligible symbols found.'); return
    state=load_state(); signals=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures={pool.submit(analyze_symbol,s):s for s in symbols}
        for future in as_completed(futures):
            symbol=futures[future]
            try:
                result=future.result()
                if result:signals.append(result); log.info('Confirmed candidate: %s %s | Score=%s/12',symbol,result['side'],result['score'])
                else:log.info('No final confirmation: %s',symbol)
            except Exception as exc:log.exception('Worker failed for %s: %s',symbol,exc)
    signals.sort(key=lambda x:x['score'],reverse=True); sent=0
    for s in signals:
        key=f"{s['symbol']}:{s['side']}"; candle=str(s['candle_ts'])
        if state.get(key)==candle:log.info('Duplicate signal skipped: %s %s',s['symbol'],s['side']); continue
        if send_telegram(build_message(s)):
            state[key]=candle; save_state(state); sent+=1
            log.info('Telegram signal sent: %s %s',s['symbol'],s['side'])
    log.info('Scan complete | Available symbols scanned=%s | Confirmed=%s | Sent=%s',len(symbols),len(signals),sent)

if __name__=='__main__':
    main()
