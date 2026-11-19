import os, json, time, logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np

# MASTER SMC + PUBLIC ORDER FLOW ALERT BOT
# Bitget USDT-M Futures | 10 crypto + XAUUSDT if contract is listed
# 4H/1H trend -> 15M structure/liquidity -> 5M entry confirmation
# Signals only. No orders are ever placed.
BASE='https://api.bitget.com'; PRODUCT='USDT-FUTURES'
TOKEN=os.getenv('TELEGRAM_BOT_TOKEN','').strip(); CHAT=os.getenv('TELEGRAM_CHAT_ID','').strip()
STATE=os.getenv('STATE_FILE','bot_state.json'); WORKERS=3; LIMIT=200
CRYPTO=['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','LTCUSDT']
GOLD='XAUUSDT'; MS={'5m':300000,'15m':900000,'1H':3600000,'4H':14400000}
MIN_SCORE=8; MIN_VOL=1.20; MIN_BODY=.50; SEND_DIAG=os.getenv('SEND_DIAGNOSTICS','true').lower()=='true'
logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO').upper(),format='%(asctime)s | %(levelname)s | %(message)s')
log=logging.getLogger('MASTER-SMC'); S=requests.Session(); S.headers['User-Agent']='Master-SMC-OrderFlow/3.1'

def get(url,params=None):
    err=None
    for n in range(3):
        try:
            r=S.get(url,params=params,timeout=20)
            if r.status_code==429 or r.status_code>=500: raise RuntimeError(f'HTTP {r.status_code}: {r.text[:150]}')
            r.raise_for_status(); j=r.json()
            if str(j.get('code'))!='00000': raise RuntimeError(f"API {j.get('code')}: {j.get('msg')}")
            return j.get('data',[])
        except Exception as e:
            err=e
            if n<2: time.sleep(2**n)
    raise RuntimeError(f'GET failed: {err}')

def tg(method,params=None):
    if not TOKEN: raise RuntimeError('Missing GitHub Secret TELEGRAM_BOT_TOKEN')
    r=S.get(f'https://api.telegram.org/bot{TOKEN}/{method}',params=params or {},timeout=20); r.raise_for_status(); j=r.json()
    if not j.get('ok'): raise RuntimeError(f"Telegram {method}: {j.get('description','unknown error')}")
    return j.get('result')

def telegram_test():
    if not CHAT: raise RuntimeError('Missing GitHub Secret TELEGRAM_CHAT_ID')
    me=tg('getMe'); chat=tg('getChat',{'chat_id':CHAT})
    log.info('Telegram connection OK: @%s; chat type=%s',me.get('username'),chat.get('type'))

def send(msg):
    if not TOKEN or not CHAT: log.error('Telegram credentials missing'); return False
    parts=[]; cur=''
    for line in msg.splitlines():
        if len(cur)+len(line)+1>3800: parts.append(cur); cur=''
        cur += ('\n' if cur else '')+line
    if cur: parts.append(cur)
    try:
        for p in parts: tg('sendMessage',{'chat_id':CHAT,'text':p,'disable_web_page_preview':'true'})
        return True
    except Exception as e: log.error('Telegram send failed: %s',e); return False

def universe():
    cs=get(BASE+'/api/v2/mix/market/contracts',{'productType':PRODUCT}); cmap={str(c.get('symbol','')).upper():c for c in cs}
    out=[]
    for sym in CRYPTO+[GOLD]:
        c=cmap.get(sym)
        if not c: log.error('Contract not listed: %s',sym); continue
        st=str(c.get('symbolStatus',c.get('status','normal'))).lower()
        if st and st not in ('normal','online','listed','trading'): log.warning('Not trading %s status=%s',sym,st); continue
        out.append(sym)
    if GOLD not in out: log.warning('XAUUSDT unavailable on this Bitget product; Gold scan skipped.')
    return out

def candles(sym,tf):
    rows=get(BASE+'/api/v2/mix/market/history-candles',{'symbol':sym,'productType':PRODUCT,'granularity':tf,'limit':str(LIMIT),'endTime':str(int(time.time()*1000))})
    if not rows: raise RuntimeError(f'No {tf} candles')
    a=[]
    for r in rows:
        try: a.append([int(r[0]),*map(float,r[1:7])])
        except (ValueError,TypeError,IndexError): pass
    d=pd.DataFrame(a,columns=['ts','open','high','low','close','volume','turnover']).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    d=d[d.ts+MS[tf]<=int(time.time()*1000)].copy()
    if len(d)<80: raise RuntimeError(f'Insufficient closed candles: {len(d)}')
    for n in (9,21,50,200): d[f'e{n}']=d.close.ewm(span=n,adjust=False).mean()
    delta=d.close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    rs=gain.ewm(alpha=1/14,adjust=False).mean()/loss.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
    d['rsi']=(100-100/(1+rs)).fillna(50)
    pc=d.close.shift(); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d['atr']=tr.ewm(alpha=1/14,adjust=False).mean(); up=d.high.diff(); dn=-d.low.diff()
    p=up.where((up>dn)&(up>0),0).ewm(alpha=1/14,adjust=False).mean(); m=dn.where((dn>up)&(dn>0),0).ewm(alpha=1/14,adjust=False).mean()
    pdi=100*p/d.atr.replace(0,np.nan); mdi=100*m/d.atr.replace(0,np.nan); dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d['adx']=dx.ewm(alpha=1/14,adjust=False).mean().fillna(0); d['vr']=(d.volume/d.volume.rolling(20).mean().replace(0,np.nan)).fillna(0)
    d['br']=((d.close-d.open).abs()/(d.high-d.low).replace(0,np.nan)).fillna(0)
    return d

def trend(d):
    x=d.iloc[-1]
    if x.close>x.e50>x.e200 and x.e9>x.e21:return 'BULLISH'
    if x.close<x.e50<x.e200 and x.e9<x.e21:return 'BEARISH'
    return 'MIXED'

def pivots(d,l=3,r=3):
    hs=[]; ls=[]
    for i in range(l,len(d)-r):
        if d.high.iloc[i]>=d.high.iloc[i-l:i+r+1].max():hs.append((i,float(d.high.iloc[i])))
        if d.low.iloc[i]<=d.low.iloc[i-l:i+r+1].min():ls.append((i,float(d.low.iloc[i])))
    return hs,ls

def structure(d):
    hs,ls=pivots(d); x=d.iloc[-1]
    if len(hs)<2 or len(ls)<2:return {'bias':'RANGE','bos':None,'choch':None,'high':float(d.high.iloc[-31:-1].max()),'low':float(d.low.iloc[-31:-1].min())}
    h1,h2=hs[-2][1],hs[-1][1]; l1,l2=ls[-2][1],ls[-1][1]
    old='BULLISH' if h2>h1 and l2>l1 else 'BEARISH' if h2<h1 and l2<l1 else 'RANGE'
    bu=float(x.close)>h2; bd=float(x.close)<l2; bias='BULLISH' if bu or (old=='BULLISH' and not bd) else 'BEARISH' if bd or (old=='BEARISH' and not bu) else 'RANGE'
    return {'bias':bias,'bos':'UP' if bu else 'DOWN' if bd else None,'choch':'UP' if old=='BEARISH' and bu else 'DOWN' if old=='BULLISH' and bd else None,'high':h2,'low':l2}

def liquidity(d):
    hs,ls=pivots(d); x=d.iloc[-1]; ph=max([v for _,v in hs[-5:]],default=float(d.high.iloc[-31:-1].max())); pl=min([v for _,v in ls[-5:]],default=float(d.low.iloc[-31:-1].min())); tol=float(x.close)*.001
    return {'sweep_high':x.high>ph and x.close<ph,'sweep_low':x.low<pl and x.close>pl,'eqh':sum(abs(v-ph)<=tol for _,v in hs[-8:])>=2,'eql':sum(abs(v-pl)<=tol for _,v in ls[-8:])>=2}

def pd_zone(d):
    w=d.iloc[-50:]; mid=(float(w.high.max())+float(w.low.min()))/2; p=float(d.close.iloc[-1])
    return 'DISCOUNT' if p<mid else 'PREMIUM' if p>mid else 'EQUILIBRIUM'

def fvg(d):
    a,m,c=d.iloc[-3],d.iloc[-2],d.iloc[-1]
    if c.low>a.high and m.close>m.open:return {'type':'BULLISH','low':float(a.high),'high':float(c.low)}
    if c.high<a.low and m.close<m.open:return {'type':'BEARISH','low':float(c.high),'high':float(a.low)}
    return None

def order_block(d,side):
    x=d.iloc[-1]; recent=d.iloc[-13:-1]; atr=max(float(x.atr),1e-12)
    if abs(x.close-x.open)<.8*atr or x.br<.55:return None
    if side=='LONG' and x.close>x.open and x.close>recent.high.max():opp=recent[recent.close<recent.open]
    elif side=='SHORT' and x.close<x.open and x.close<recent.low.min():opp=recent[recent.close>recent.open]
    else:return None
    if opp.empty:return None
    z=opp.iloc[-1]; return {'low':float(z.low),'high':float(z.high)}

def bpr(d):
    bull=[]; bear=[]
    for i in range(3,min(35,len(d)-1)):
        a,c=d.iloc[-i-1],d.iloc[-i+1]
        if c.low>a.high:bull.append((float(a.high),float(c.low)))
        if c.high<a.low:bear.append((float(c.high),float(a.low)))
    for a in bull:
        for z in bear:
            lo=max(a[0],z[0]); hi=min(a[1],z[1])
            if lo<hi:return {'low':lo,'high':hi}
    return None

def flow(sym):
    book=get(BASE+'/api/v2/mix/market/merge-depth',{'symbol':sym,'productType':PRODUCT,'precision':'scale0'})
    fills=get(BASE+'/api/v2/mix/market/fills',{'symbol':sym,'productType':PRODUCT,'limit':'100'})
    if not isinstance(book,dict):raise RuntimeError('Unexpected order-book payload')
    b=sum(float(z[1]) for z in book.get('bids',[])[:20]); a=sum(float(z[1]) for z in book.get('asks',[])[:20]); total=b+a
    buy=sell=0.0
    for t in fills if isinstance(fills,list) else []:
        q=float(t.get('size',0) or 0)
        if str(t.get('side','')).lower()=='buy':buy+=q
        elif str(t.get('side','')).lower()=='sell':sell+=q
    tt=buy+sell
    return {'imbalance':(b-a)/total if total else 0,'delta':(buy-sell)/tt if tt else 0}

def analyze(sym):
    try:
        d4=candles(sym,'4H'); d1=candles(sym,'1H'); d15=candles(sym,'15m'); d5=candles(sym,'5m'); t4=trend(d4); t1=trend(d1)
        side='LONG' if t4==t1=='BULLISH' else 'SHORT' if t4==t1=='BEARISH' else None
        if not side:return None,f'4H/1H trend not aligned (4H={t4}, 1H={t1})'
        st=structure(d15)
        if st['bias']!=('BULLISH' if side=='LONG' else 'BEARISH'):return None,f'15M structure/BOS not aligned ({st["bias"]})'
        liq=liquidity(d15); pdz=pd_zone(d15); gap=fvg(d15); ob=order_block(d15,side); balance=bpr(d15); of=flow(sym); x=d5.iloc[-1]; prev=d5.iloc[-2]
        if side=='LONG': candle=x.close>x.open and x.close>prev.high
        else:candle=x.close<x.open and x.close<prev.low
        if not candle or x.br<MIN_BODY or abs(x.close-x.open)<.45*max(x.atr,1e-12) or x.vr<MIN_VOL:return None,'5M closed candle/body/volume confirmation failed'
        start=float(d5.close.iloc[-7]); move=(float(x.close)-start)/start*100 if start else 0; atrpct=float(x.atr)/float(x.close)*100
        if (side=='LONG' and move<max(.25,atrpct*.8)) or (side=='SHORT' and move>-max(.25,atrpct*.8)):return None,'5M displacement/momentum not strong enough'
        aligned=(of['imbalance']>=.05 and of['delta']>=.02) if side=='LONG' else (of['imbalance']<=-.05 and of['delta']<=-.02)
        swept=liq['sweep_low'] if side=='LONG' else liq['sweep_high']; pdok=pdz==('DISCOUNT' if side=='LONG' else 'PREMIUM')
        gapok=bool(gap and gap['type']==('BULLISH' if side=='LONG' else 'BEARISH'))
        score=6+int(aligned)+int(swept)+int(pdok)+int(gapok)+int(bool(ob))+int(float(x.adx)>=16)
        if score<MIN_SCORE:return None,f'Confluence score {score}/12 below {MIN_SCORE}; flow={aligned}, sweep={swept}, PD={pdok}'
        if abs(float(x.close)-float(x.e21))>2.8*max(float(x.atr),1e-12):return None,'5M price overextended from EMA21'
        entry=float(x.close); atr=max(float(x.atr),entry*.0005)
        if side=='LONG':sl=min(entry-1.4*atr,(ob['low']-.15*atr) if ob else entry-1.4*atr); risk=entry-sl; lo,hi=entry-.2*atr,entry+.1*atr; tps=[entry+risk,entry+2*risk,entry+3*risk]
        else:sl=max(entry+1.4*atr,(ob['high']+.15*atr) if ob else entry+1.4*atr); risk=sl-entry; lo,hi=entry-.1*atr,entry+.2*atr; tps=[entry-risk,entry-2*risk,entry-3*risk]
        return {'symbol':sym,'side':side,'score':score,'entry':entry,'lo':lo,'hi':hi,'sl':sl,'tps':tps,'ts':int(x.ts),'t4':t4,'t1':t1,'st':st,'liq':liq,'pd':pdz,'gap':gap,'ob':ob,'bpr':balance,'flow':of},'CONFIRMED'
    except Exception as e:
        log.exception('Analysis error %s',sym); return None,f'API/analysis error: {str(e)[:160]}'

def load_state():
    try:
        with open(STATE,'r',encoding='utf-8') as f:return json.load(f)
    except FileNotFoundError:return {}
    except Exception as e:log.warning('State read error: %s',e); return {}

def save_state(s):
    tmp=STATE+'.tmp'
    with open(tmp,'w',encoding='utf-8') as f:json.dump(s,f,indent=2)
    os.replace(tmp,STATE)

def fmt(v):
    v=float(v)
    return f'{v:,.2f}' if v>=1000 else (f'{v:.6f}'.rstrip('0').rstrip('.') if v>=1 else f'{v:.8f}'.rstrip('0').rstrip('.'))

def message(s):
    f=s['flow']; return (f"✅ FINAL CONFIRMED {'🟢 LONG' if s['side']=='LONG' else '🔴 SHORT'}\nSymbol: {s['symbol']} | Score: {s['score']}/12\n\nEntry Area: {fmt(s['lo'])} - {fmt(s['hi'])}\nEntry: {fmt(s['entry'])}\nStop Loss: {fmt(s['sl'])}\nTP1: {fmt(s['tps'][0])}\nTP2: {fmt(s['tps'][1])}\nTP3: {fmt(s['tps'][2])}\n\n4H: {s['t4']} | 1H: {s['t1']}\n15M: {s['st']}\nLiquidity: {s['liq']} | PD: {s['pd']}\nFVG: {s['gap']} | OB: {s['ob']} | BPR: {s['bpr']}\nOrder-book imbalance: {f['imbalance']:+.3f}\nRecent-trade delta proxy: {f['delta']:+.3f}\n\nClosed 5M confirmation. Alert only; not guaranteed.")

def main():
    log.info('MASTER SMC BOT START')
    telegram_test()  # getMe + getChat; no test message is sent
    syms=universe()
    if not syms:raise RuntimeError('No available configured contracts')
    state=load_state(); reasons={}; signals=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fs={pool.submit(analyze,s):s for s in syms}
        for fu in as_completed(fs):
            sym=fs[fu]
            try:s,why=fu.result()
            except Exception as e:s,why=None,f'Worker failure: {e}'
            reasons[sym]=why
            if s:signals.append(s); log.info('CONFIRMED %s %s score=%s',sym,s['side'],s['score'])
            else:log.info('NO TRADE %s: %s',sym,why)
    sent=0
    for s in sorted(signals,key=lambda z:z['score'],reverse=True):
        k=f"{s['symbol']}:{s['side']}"; ck=str(s['ts'])
        if state.get(k)==ck:log.info('Duplicate blocked %s',k); continue
        if send(message(s)):state[k]=ck;save_state(state);sent+=1
    log.info('Scan complete: scanned=%d confirmed=%d sent=%d',len(syms),len(signals),sent)
    if SEND_DIAG:
        lines=['📊 SMC BOT SCAN DIAGNOSTICS',f"UTC: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",f'Scanned={len(syms)} Confirmed={len(signals)} Sent={sent}','']
        lines += [f"{'✅' if reasons.get(s)=='CONFIRMED' else '⛔'} {s}: {reasons.get(s,'No result')}" for s in syms]
        send('\n'.join(lines))

if __name__=='__main__': main()
