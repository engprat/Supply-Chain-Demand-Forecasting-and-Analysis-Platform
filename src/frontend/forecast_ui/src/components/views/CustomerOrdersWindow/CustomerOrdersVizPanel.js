// CustomerOrdersVizPanel.js
import React from "react";
import { motion } from "framer-motion";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  PieChart, Pie, Cell, Legend
} from "recharts";
import "./CustomerOrdersVizPanel.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

/* ------------------------------ helpers ------------------------------ */
async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
function toISO(d) {
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0,10);
}
function daysBetween(aISO, bISO){ const a=new Date(aISO), b=new Date(bISO); return Math.max(1, Math.floor((b-a)/(24*3600*1000))+1); }
function fmt(n,d=0){ return n==null||Number.isNaN(n) ? "—" : Number(n).toLocaleString(undefined,{maximumFractionDigits:d}); }
function truncate(s, max=18){ if(!s) return ""; return s.length>max ? s.slice(0,max-1)+"…" : s; }

const COLORS = [
  "#4EA1FF","#5AD07A","#FFB84E","#FF6B6B","#A78BFA","#50E3C2",
  "#F45B69","#7FDBDA","#F7A072","#9DB0D8","#FFD166","#06D6A0",
  "#EF476F","#8EC07C","#61AFEF","#E5C07B","#56B6C2","#D19A66"
];

/* -------------------------- mini typeahead --------------------------- */
function useDebounced(value, ms=300){
  const [v,setV]=React.useState(value);
  React.useEffect(()=>{ const t=setTimeout(()=>setV(value),ms); return ()=>clearTimeout(t); },[value,ms]);
  return v;
}

function OptionsSelect({ label, column, picked, setPicked, limit=50, placeholder="Type to search…" }) {
  const [q,setQ]=React.useState("");
  const [open,setOpen]=React.useState(false);
  const deb=useDebounced(q,300);
  const [opts,setOpts]=React.useState([]);
  const [loading,setLoading]=React.useState(false);
  const [err,setErr]=React.useState("");

  React.useEffect(()=>{
    let live=true;
    (async()=>{
      try{
        setLoading(true); setErr("");
        const url = new URL(`${API_BASE}/api/customer-orders/options`);
        url.searchParams.set("column",column);
        url.searchParams.set("limit",String(limit));
        if (deb) url.searchParams.set("q",deb);
        const data = await fetchJSON(url);
        const vals = (data?.[column] || []); // cap when empty query
        if (live) setOpts(vals);
      }catch(e){ if(live) setErr("Lookup failed"); }
      finally{ if(live) setLoading(false); }
    })();
    return ()=>{ live=false; };
  },[column,deb,limit]);

  const add = (v)=>{ if(!v||picked.includes(v))return; setPicked([...picked,v]); setQ(""); };
  const remove = (v)=> setPicked(picked.filter(x=>x!==v));

  return (
    <div className="co-field">
      <label>{label}</label>
      {!!picked.length && (
        <div className="co-chips">
          {picked.map(v=>(
            <span className="co-chip" key={v} title={v}>
              {truncate(v,22)}
              <button onClick={()=>remove(v)} aria-label="remove">×</button>
            </span>
          ))}
          <button className="co-chip-clear" onClick={()=>setPicked([])}>Clear</button>
        </div>
      )}
      <div className="co-input-wrap">
        <input
          value={q}
          onFocus={()=>setOpen(true)}
          onBlur={()=>setTimeout(()=>setOpen(false),150)}
          onChange={e=>setQ(e.target.value)}
          placeholder={placeholder}
          aria-label={`${label} search`}
        />
        {loading && <span className="co-spinner" aria-hidden>⏳</span>}
      </div>
      {err && <div className="co-err">{err}</div>}

      {open && (!!opts.length) && (
        <div className="co-options" role="listbox">
          {opts.map(v=> (
            <div className="co-option" key={v} onMouseDown={()=>add(v)} title={v}>
              {truncate(v)}
            </div>
          ))}
          {!deb && opts.length>=20 && (
            <div className="co-options-foot">Type to narrow results…</div>
          )}
        </div>
      )}
    </div>
  );
}

/* ----------------------- custom recharts bits ------------------------ */
// Active pie slice that pulls out slightly on hover/tap
function renderActiveSlice(props){
  const RAD = Math.PI / 180;
  const { cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle, fill, payload, value } = props;
  const offset = 8;
  const sx = cx + Math.cos(-midAngle * RAD) * offset;
  const sy = cy + Math.sin(-midAngle * RAD) * offset;

  return (
    <g>
      <defs>
        <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="2" result="blur"/>
          <feOffset in="blur" dx="0" dy="1" result="offsetBlur"/>
          <feMerge><feMergeNode in="offsetBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <text x={sx} y={sy} dy={0} textAnchor="middle" fill="#e8eefc" fontSize="12" filter="url(#dropShadow)">
        {truncate(payload?.name,18)}
      </text>
      <SectorLike
        cx={sx} cy={sy}
        innerRadius={innerRadius} outerRadius={outerRadius+4}
        startAngle={startAngle} endAngle={endAngle}
        fill={fill}
      />
      <text x={sx} y={sy+14} dy={8} textAnchor="middle" fill="#9db0d8" fontSize="11">
        {fmt(value)}
      </text>
    </g>
  );
}
function SectorLike({ cx,cy,innerRadius,outerRadius,startAngle,endAngle,fill }){
  return (
    <path
      d={makeDonutArcPath(cx, cy, innerRadius, outerRadius, startAngle, endAngle)}
      fill={fill}
      stroke="rgba(0,0,0,0.15)"
      strokeWidth="1"
    />
  );
}
function makeDonutArcPath(cx,cy,ir,or,start,end){
  const toRad = (a)=> (Math.PI/180)*a;
  const sx = cx + or*Math.cos(toRad(-start));
  const sy = cy + or*Math.sin(toRad(-start));
  const ex = cx + or*Math.cos(toRad(-end));
  const ey = cy + or*Math.sin(toRad(-end));
  const large = (end-start)%360 > 180 ? 1 : 0;

  const six = cx + ir*Math.cos(toRad(-end));
  const siy = cy + ir*Math.sin(toRad(-end));
  const sxx = cx + ir*Math.cos(toRad(-start));
  const sxy = cy + ir*Math.sin(toRad(-start));

  return [
    `M ${sx} ${sy}`,
    `A ${or} ${or} 0 ${large} 0 ${ex} ${ey}`,
    `L ${six} ${siy}`,
    `A ${ir} ${ir} 0 ${large} 1 ${sxx} ${sxy}`,
    "Z"
  ].join(" ");
}

function ActiveDot(props){
  const { cx, cy } = props;
  return (
    <g>
      <circle cx={cx} cy={cy} r={6} stroke="#fff" strokeWidth={2} fill="none"/>
      <motion.circle
        cx={cx} cy={cy} r={4}
        initial={{ scale: 0.9, opacity: 0.6 }}
        animate={{ scale: [1, 1.2, 1], opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
        fill="#4EA1FF"
      />
    </g>
  );
}

/* ----------------------------- main panel ---------------------------- */
export default function CustomerOrdersVizPanel(){
  // default date window (last 14 days)
  const today = new Date();
  const d14 = new Date(today); d14.setDate(today.getDate() - 13);
  const [start,setStart]=React.useState(toISO(d14));
  const [end,setEnd]=React.useState(toISO(today));

  const [customers,setCustomers]=React.useState([]);
  const [products,setProducts]=React.useState([]);
  const [cities,setCities]=React.useState([]);

  const [minQty,setMinQty]=React.useState("");
  const [minPct,setMinPct]=React.useState("");

  const [loading,setLoading]=React.useState(false);
  const [err,setErr]=React.useState("");

  const [rows,setRows]=React.useState([]);
  const [stats,setStats]=React.useState(null);

  // animation reset key (bump to retrigger initial render animations)
  const [animKey,setAnimKey]=React.useState(0);

  // pie hover/tap focus
  const [activeCityIdx,setActiveCityIdx]=React.useState(-1);
  const [activeCustIdx,setActiveCustIdx]=React.useState(-1);
  const [activeProdIdx,setActiveProdIdx]=React.useState(-1);

  // promo overlay (optional)
  const [channel,setChannel]=React.useState("");
  const [promoDays,setPromoDays]=React.useState(null);

  const areaData = React.useMemo(()=> {
    const map = new Map();
    for (const r of rows){
      const d = r.date;
      const q = Number(r.pred_order_qty)||0;
      map.set(d,(map.get(d)||0)+q);
    }
    return Array.from(map.entries()).sort((a,b)=>a[0].localeCompare(b[0]))
      .map(([date,qty])=>({ date, qty }));
  },[rows]);

  const pies = React.useMemo(()=>{
    const agg = (key) => {
      const m = new Map();
      for(const r of rows){
        const k = (r[key] ?? "Unknown") || "Unknown";
        const q = Number(r.pred_order_qty)||0;
        m.set(k,(m.get(k)||0)+q);
      }
      const all = Array.from(m.entries()).map(([name,value])=>({name,value})).sort((a,b)=>b.value-a.value);
      const top = all.slice(0,12);
      if (all.length>12){
        const others = all.slice(12).reduce((s,x)=>s+x.value,0);
        top.push({name:"Others",value:others});
      }
      return top;
    };
    return {
      byCity: agg("city"),
      byCustomer: agg("customer_id"),
      byProduct: agg("product_id"),
      total: rows.reduce((s,r)=>s+(Number(r.pred_order_qty)||0),0)
    };
  },[rows]);

  async function loadData(){
    try{
      setLoading(true); setErr("");
      const url = new URL(`${API_BASE}/api/customer-orders/predict`);
      url.searchParams.set("page","1");
      url.searchParams.set("page_size","5000");
      url.searchParams.set("sort_by","date");
      url.searchParams.set("sort_dir","asc");
      if (minPct) url.searchParams.set("min_pct", String(minPct));

      const body = {
        start, end,
        customers: customers.length ? customers : null,
        products:  products.length ? products  : null,
        cities:    cities.length    ? cities    : null,
        min_qty:   minQty ? Number(minQty) : null
      };

      const data = await fetchJSON(url, {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify(body)
      });
      setRows(data?.predictions || []);
      setStats(data?.stats || null);
      setAnimKey(k=>k+1);

      if (products.length===1 && (cities.length===1 || channel)){
        const horizon = daysBetween(start,end);
        const payload = {
          sku_id: products[0],
          location: cities[0] || null,
          channel: channel || null,
          forecast_days: horizon,
          region: "USA",
          forecast_granularity: "daily",
          promo_modeling_approach: "separated"
        };
        try{
          const promo = await fetchJSON(`${API_BASE}/api/promotional-forecast`,{
            method:"POST",
            headers:{ "Content-Type":"application/json" },
            body: JSON.stringify(payload)
          });
          const active = (promo?.promotional_demand_forecast||[]).map(x=>Number(x)||0).filter(x=>x>0).length;
          setPromoDays(active);
        }catch{ setPromoDays(null); }
      }else{
        setPromoDays(null);
      }
    }catch(e){
      setErr(String(e.message || e));
      setRows([]); setStats(null); setPromoDays(null);
    }finally{
      setLoading(false);
    }
  }

  const totalUnits = React.useMemo(()=> areaData.reduce((s,x)=>s+x.qty,0),[areaData]);
  const horizonDays = React.useMemo(()=> areaData.length, [areaData]);
  const avgPerDay = React.useMemo(()=> horizonDays? totalUnits/horizonDays : 0, [totalUnits,horizonDays]);

  /* ----------------------------- rendering ---------------------------- */
  return (
    <div className="co-panel">
      <header className="co-header">
        <h2>Customer Orders (ML) — Animated Insights</h2>
        <div className="co-sub">Area trend + comparative pies with hover focus &amp; tooltips</div>
        <div className="co-window">Window: {start} → {end}</div>
      </header>

      <section className="co-filters">
        <div className="co-row">
          <div className="co-field"><label>Start</label><input type="date" value={start} onChange={e=>setStart(e.target.value)}/></div>
          <div className="co-field"><label>End</label><input type="date" value={end} onChange={e=>setEnd(e.target.value)}/></div>
          <div className="co-field"><label>Min Qty</label><input type="number" min="0" value={minQty} onChange={e=>setMinQty(e.target.value)}/></div>
          <div className="co-field"><label>Min Percentile (0–100)</label><input type="number" min="0" max="100" value={minPct} onChange={e=>setMinPct(e.target.value)}/></div>
        </div>
        <div className="co-row">
          <OptionsSelect label="Customers" column="customer_id" picked={customers} setPicked={setCustomers}/>
          <OptionsSelect label="Products"  column="product_id" picked={products}  setPicked={setProducts}/>
          <OptionsSelect label="Cities"    column="city"       picked={cities}    setPicked={setCities}/>
          <div className="co-field">
            <label>Channel (for promo overlay)</label>
            <input placeholder="e.g., E-commerce" value={channel} onChange={e=>setChannel(e.target.value)}/>
          </div>
        </div>
        <div className="co-actions">
          <button className="primary" disabled={loading || !products.length || !customers.length} onClick={loadData}>
            {loading ? "Loading…" : "Load Charts"}
          </button>
          {err && <span className="co-err co-banner">{err}</span>}
          {!customers.length || !products.length ? (
            <span className="co-hint-inline">Pick at least one customer and one product to enable charts.</span>
          ) : null}
        </div>
      </section>

      {/* KPI strip */}
      <section className="co-kpis">
        <div className="co-kpi">
          <div className="co-kpi-label">Total Units</div>
          <div className="co-kpi-value">{fmt(totalUnits)}</div>
          <div className="co-kpi-sub">{horizonDays} days</div>
        </div>
        <div className="co-kpi">
          <div className="co-kpi-label">Avg / Day</div>
          <div className="co-kpi-value">{fmt(avgPerDay,1)}</div>
          <div className="co-kpi-sub">post-filter</div>
        </div>
        <div className="co-kpi">
          <div className="co-kpi-label">Median • p90 • p95</div>
          <div className="co-kpi-value">
            {fmt(stats?.median,1)} • {fmt(stats?.p90,1)} • {fmt(stats?.p95,1)}
          </div>
          <div className="co-kpi-sub">from stats</div>
        </div>
        <div className="co-kpi">
          <div className="co-kpi-label">Promo Days (horizon)</div>
          <div className="co-kpi-value">{promoDays==null ? "—" : fmt(promoDays)}</div>
          <div className="co-kpi-sub">best-effort overlay</div>
        </div>
      </section>

      {/* Area chart with progressive reveal */}
      <section className="co-card">
        <h3>Time Series — Area Trend</h3>
        <div className="co-chart-wrap">
          <motion.div
            key={`reveal-${animKey}`}
            className="co-reveal"
            initial={{ width: 0 }}
            animate={{ width: "100%" }}
            transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
          >
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart
                data={areaData}
                margin={{ top: 10, right: 20, bottom: 0, left: 0 }}
              >
                <defs>
                  <linearGradient id="gradArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4EA1FF" stopOpacity={0.5}/>
                    <stop offset="100%" stopColor="#4EA1FF" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false}/>
                <XAxis dataKey="date" tick={{ fill: "#9db0d8", fontSize: 12 }}/>
                <YAxis tick={{ fill: "#9db0d8", fontSize: 12 }}/>
                <Tooltip
                  contentStyle={{ background:"#0f1728", border:"1px solid #223150", color:"#e8eefc" }}
                  labelStyle={{ color:"#9db0d8" }}
                  formatter={(v)=>[fmt(v),"Units"]}
                />
                <Area
                  type="monotone"
                  dataKey="qty"
                  stroke="#4EA1FF"
                  strokeWidth={2}
                  fill="url(#gradArea)"
                  isAnimationActive={true}
                  animationBegin={100}
                  animationDuration={1200}
                  activeDot={<ActiveDot />}
                />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>
        </div>
        <div className="co-hint">
          Initial render “draws” the line from left→right while the area fades in. Hover shows a pulsing point.
        </div>
      </section>

      {/* Comparison pies */}
      <section className="co-grid">
        <PieCard
          title="By Customer"
          data={pies.byCustomer}
          total={pies.total}
          activeIndex={activeCustIdx}
          setActiveIndex={setActiveCustIdx}
          colorOffset={3}
        />
        <PieCard
          title="By Product"
          data={pies.byProduct}
          total={pies.total}
          activeIndex={activeProdIdx}
          setActiveIndex={setActiveProdIdx}
          colorOffset={6}
        />
        <PieCard
          title="By City"
          data={pies.byCity}
          total={pies.total}
          activeIndex={activeCityIdx}
          setActiveIndex={setActiveCityIdx}
          colorOffset={0}
        />
      </section>
    </div>
  );
}

/* ---------------------------- sub components ------------------------- */
function PieCard({ title, data, total, activeIndex, setActiveIndex, colorOffset=0 }){
  const has = data && data.length>0;
  return (
    <div className="co-card co-pie-card">
      <div className="co-card-head">
        <h3>{title}</h3>
        <div className="co-card-sub">Hover or tap a slice for details</div>
      </div>

      <div className="co-pie-wrap">
        {!has ? (
          <div className="co-empty">No data. Load charts with selections.</div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie
                data={data}
                dataKey="value" nameKey="name"
                cx="42%" cy="50%" innerRadius={60} outerRadius={100}
                paddingAngle={2}
                isAnimationActive={true}
                animationBegin={150}
                animationDuration={1100}
                activeIndex={activeIndex}
                onMouseEnter={(_,i)=>setActiveIndex(i)}
                onMouseLeave={()=>setActiveIndex(-1)}
                onTouchStart={(_,i)=>setActiveIndex(i)}
                onClick={(_,i)=>setActiveIndex(i)}
                activeShape={renderActiveSlice}
              >
                {data.map((_,i)=><Cell key={i} fill={COLORS[(i+colorOffset)%COLORS.length]} />)}
              </Pie>

              <Legend
                layout="vertical"
                verticalAlign="middle"
                align="right"
                wrapperStyle={{ color:"#9db0d8", width: 150, overflowY: "auto", maxHeight: 260 }}
                formatter={(val)=>truncate(val,18)}
              />

              <Tooltip
                contentStyle={{ background:"#0f1728", border:"1px solid #223150", color:"#e8eefc" }}
                formatter={(v, n)=>[fmt(v), n]}
                labelFormatter={(n)=>truncate(n)}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>

      {has && (
        <div className="co-caption">
          Total: {fmt(total)} units • Segments: {data.length}
        </div>
      )}
    </div>
  );
}
