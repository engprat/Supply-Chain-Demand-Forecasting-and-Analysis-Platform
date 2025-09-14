// CustomerOrdersPies.js
import React from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from "recharts";

const COLORS = [
  "#4EA1FF", "#5AD07A", "#FFB84E", "#FF6B6B", "#A78BFA", "#50E3C2",
  "#F45B69", "#7FDBDA", "#F7A072", "#9DB0D8", "#FFD166", "#06D6A0",
  "#B794F4", "#39C0ED", "#F28B82", "#FDD663", "#81C995", "#A7FFEB",
];

const toNum = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
};

const fmt = (n) =>
  n == null || Number.isNaN(n)
    ? "—"
    : Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });

function aggregate(rows, key, topN = 12) {
  const m = new Map();
  for (const r of rows) {
    const name = (r?.[key] ?? "Unknown") || "Unknown";
    const qty = toNum(
      r?.pred_order_qty ??
        r?.predicted_order_qty ??
        r?.prediction ??
        r?.qty ??
        r?.quantity
    );
    if (!m.has(name)) m.set(name, 0);
    m.set(name, m.get(name) + qty);
  }

  // Build array and drop zero-value categories
  const arr = Array.from(m.entries())
    .map(([name, value]) => ({ name, value }))
    .filter((d) => d.value > 0)
    .sort((a, b) => b.value - a.value);

  if (arr.length <= topN) return arr;

  const head = arr.slice(0, topN);
  const others = arr.slice(topN).reduce((s, d) => s + d.value, 0);
  head.push({ name: "Others", value: others });
  return head;
}

/* --------- active slice rendering: slight pull-out + labels ---------- */
function renderActiveSlice(props) {
  const RAD = Math.PI / 180;
  const {
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    startAngle,
    endAngle,
    fill,
    payload,
    value,
  } = props;

  const offset = 8;
  const sx = cx + Math.cos(-midAngle * RAD) * offset;
  const sy = cy + Math.sin(-midAngle * RAD) * offset;

  return (
    <g>
      <defs>
        <filter id="coPieShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="2" result="b" />
          <feOffset in="b" dx="0" dy="1" result="ob" />
          <feMerge>
            <feMergeNode in="ob" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <path
        d={donutPath(sx, sy, innerRadius, outerRadius + 4, startAngle, endAngle)}
        fill={fill}
        stroke="rgba(0,0,0,0.2)"
        strokeWidth="1"
        filter="url(#coPieShadow)"
      />
      <text
        x={sx}
        y={sy - 2}
        dy={0}
        textAnchor="middle"
        fill="#e8eefc"
        fontSize="12"
      >
        {payload?.name}
      </text>
      <text
        x={sx}
        y={sy + 14}
        dy={8}
        textAnchor="middle"
        fill="#9db0d8"
        fontSize="11"
      >
        {fmt(value)}
      </text>
    </g>
  );
}

function donutPath(cx, cy, ir, or, start, end) {
  const toRad = (a) => (Math.PI / 180) * a;
  const sx = cx + or * Math.cos(toRad(-start));
  const sy = cy + or * Math.sin(toRad(-start));
  const ex = cx + or * Math.cos(toRad(-end));
  const ey = cy + or * Math.sin(toRad(-end));
  const large = ((end - start) % 360) > 180 ? 1 : 0;

  const six = cx + ir * Math.cos(toRad(-end));
  const siy = cy + ir * Math.sin(toRad(-end));
  const sxx = cx + ir * Math.cos(toRad(-start));
  const sxy = cy + ir * Math.sin(toRad(-start));

  return [
    `M ${sx} ${sy}`,
    `A ${or} ${or} 0 ${large} 0 ${ex} ${ey}`,
    `L ${six} ${siy}`,
    `A ${ir} ${ir} 0 ${large} 1 ${sxx} ${sxy}`,
    "Z",
  ].join(" ");
}

const Empty = ({ label }) => (
  <div
    style={{
      height: 260,
      display: "grid",
      placeItems: "center",
      color: "#9db0d8",
      fontSize: 13,
    }}
  >
    No data for {label}. Load charts with selections.
  </div>
);

export default function CustomerOrdersPies({ rows = [], animKey = 0 }) {
  // compute date range from rows (for the small header)
  const windowText = React.useMemo(() => {
    if (!rows.length) return null;
    const dates = rows.map((r) => r.date).filter(Boolean).sort();
    return `${dates[0]} → ${dates[dates.length - 1]}`;
  }, [rows]);

  const byCustomer = React.useMemo(
    () => aggregate(rows, "customer_id"),
    [rows]
  );
  const byProduct = React.useMemo(() => aggregate(rows, "product_id"), [rows]);
  const byCity = React.useMemo(() => aggregate(rows, "city"), [rows]);

  const [activeCust, setActiveCust] = React.useState(-1);
  const [activeProd, setActiveProd] = React.useState(-1);
  const [activeCity, setActiveCity] = React.useState(-1);

  return (
    <>
      <div className="co-grid-title">
        <div className="co-sub" style={{ marginBottom: 8 }}>
          {windowText ? <>Window: <strong>{windowText}</strong></> : null}
          <span style={{ float: "right", opacity: 0.8 }}>
            Hover or tap a slice for details
          </span>
        </div>
      </div>

      {/* By Customer */}
      <div className="co-card">
        <h3>By Customer</h3>
        <div className="co-pie-wrap">
          {!byCustomer.length ? (
            <Empty label="customers" />
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  key={`cust-${animKey}`}
                  data={byCustomer}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={95}
                  paddingAngle={2}
                  isAnimationActive={true}
                  animationBegin={0}
                  animationDuration={1100}
                  activeIndex={activeCust}
                  onMouseEnter={(_, i) => setActiveCust(i)}
                  onMouseLeave={() => setActiveCust(-1)}
                  onTouchStart={(_, i) => setActiveCust(i)}
                  onClick={(_, i) => setActiveCust(i)}
                  activeShape={renderActiveSlice}
                >
                  {byCustomer.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#0f1728",
                    border: "1px solid #223150",
                    color: "#e8eefc",
                  }}
                  formatter={(v, n) => [fmt(v), n]}
                />
                <Legend wrapperStyle={{ color: "#9db0d8" }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* By Product */}
      <div className="co-card">
        <h3>By Product</h3>
        <div className="co-pie-wrap">
          {!byProduct.length ? (
            <Empty label="products" />
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  key={`prod-${animKey}`}
                  data={byProduct}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={95}
                  paddingAngle={2}
                  isAnimationActive={true}
                  animationBegin={100}
                  animationDuration={1100}
                  activeIndex={activeProd}
                  onMouseEnter={(_, i) => setActiveProd(i)}
                  onMouseLeave={() => setActiveProd(-1)}
                  onTouchStart={(_, i) => setActiveProd(i)}
                  onClick={(_, i) => setActiveProd(i)}
                  activeShape={renderActiveSlice}
                >
                  {byProduct.map((_, i) => (
                    <Cell key={i} fill={COLORS[(i + 4) % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#0f1728",
                    border: "1px solid #223150",
                    color: "#e8eefc",
                  }}
                  formatter={(v, n) => [fmt(v), n]}
                />
                <Legend wrapperStyle={{ color: "#9db0d8" }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* By City */}
      <div className="co-card">
        <h3>By City</h3>
        <div className="co-pie-wrap">
          {!byCity.length ? (
            <Empty label="cities" />
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  key={`city-${animKey}`}
                  data={byCity}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={95}
                  paddingAngle={2}
                  isAnimationActive={true}
                  animationBegin={200}
                  animationDuration={1100}
                  activeIndex={activeCity}
                  onMouseEnter={(_, i) => setActiveCity(i)}
                  onMouseLeave={() => setActiveCity(-1)}
                  onTouchStart={(_, i) => setActiveCity(i)}
                  onClick={(_, i) => setActiveCity(i)}
                  activeShape={renderActiveSlice}
                >
                  {byCity.map((_, i) => (
                    <Cell key={i} fill={COLORS[(i + 8) % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#0f1728",
                    border: "1px solid #223150",
                    color: "#e8eefc",
                  }}
                  formatter={(v, n) => [fmt(v), n]}
                />
                <Legend wrapperStyle={{ color: "#9db0d8" }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </>
  );
}
