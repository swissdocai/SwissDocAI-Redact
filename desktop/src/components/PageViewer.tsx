import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { Box, PageView } from '../api';

export interface Highlight {
  page: number;
  box: Box;
}

interface Props {
  pages: (PageView | null)[];
  imageUrl: (n: number) => string;
  mode: 'review' | 'result';
  zoom: number; // 1 = page width fits the view
  highlight: Highlight | null;
  onClick: (page: number, x: number, y: number) => void;
  onBox: (page: number, box: Box) => void;
  onVisiblePage: (page: number) => void;
}

const FILL: Record<string, string> = {
  manual: 'rgba(234, 88, 12, 0.5)',
  signature: 'rgba(124, 58, 237, 0.45)',
  qr_code: 'rgba(8, 145, 178, 0.45)',
};

function PageCanvas({ n, page, imageUrl, width, review, highlight, onClick, onBox }: {
  n: number; page: PageView; imageUrl: string; width: number; review: boolean; highlight: Box | null;
  onClick: (page: number, x: number, y: number) => void; onBox: (page: number, box: Box) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const highlightRef = useRef<SVGRectElement>(null);
  const [drag, setDrag] = useState<{ start: [number, number]; end: [number, number]; client: [number, number] } | null>(null);

  useEffect(() => {
    highlightRef.current?.scrollIntoView({ block: 'center', inline: 'center', behavior: 'smooth' });
  }, [highlight]);

  const toPage = (event: React.MouseEvent): [number, number] => {
    const rect = svg.current!.getBoundingClientRect();
    return [((event.clientX - rect.left) * page.width) / rect.width, ((event.clientY - rect.top) * page.height) / rect.height];
  };
  const down = (event: React.MouseEvent) => {
    if (!review || event.button !== 0) return;
    const point = toPage(event);
    setDrag({ start: point, end: point, client: [event.clientX, event.clientY] });
  };
  const move = (event: React.MouseEvent) => {
    if (drag) setDrag({ ...drag, end: toPage(event) });
  };
  const up = (event: React.MouseEvent) => {
    if (!drag) return;
    const moved = Math.hypot(event.clientX - drag.client[0], event.clientY - drag.client[1]);
    const [x0, y0] = drag.start;
    const [x1, y1] = toPage(event);
    setDrag(null);
    if (moved < 5) onClick(n, x0, y0);
    else onBox(n, [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]);
  };

  const stroke = Math.max(2, page.width / 600);
  const rect = (box: Box, pad: number) => ({
    x: box[0] - pad, y: box[1] - pad, width: box[2] - box[0] + 2 * pad, height: box[3] - box[1] + 2 * pad,
  });
  return (
    <div
      className="relative overflow-hidden rounded-sm bg-white shadow-[0_2px_24px_rgba(28,25,23,.15)] ring-1 ring-stone-200"
      style={{ width, aspectRatio: `${page.width} / ${page.height}` }}
    >
      <img src={imageUrl} alt="" className="block w-full" loading="lazy" draggable={false} />
      <svg
        ref={svg}
        viewBox={`0 0 ${page.width} ${page.height}`}
        preserveAspectRatio="none"
        className={`absolute inset-0 h-full w-full ${review ? 'cursor-crosshair' : ''}`}
        onMouseDown={down}
        onMouseMove={move}
        onMouseUp={up}
        onMouseLeave={() => setDrag(null)}
      >
        {review && page.words.map((word, i) => (
          <rect key={`w${i}`} {...rect(word.box, 0)} rx={stroke}
            className="cursor-pointer fill-transparent transition-colors hover:fill-brand-500/20">
            <title>{word.text}</title>
          </rect>
        ))}
        {page.boxes.map(({ box, category }, i) => (
          <rect key={`b${i}`} {...rect(box, 0)} rx={review ? stroke : 0}
            fill={review ? FILL[category] ?? 'rgba(15, 23, 42, 0.62)' : '#000'} className="pointer-events-none" />
        ))}
        {review && [...page.words.filter((word) => word.state === 'kept'), ...page.hidden].map(({ box }, i) => (
          <rect key={`k${i}`} {...rect(box, stroke)} rx={stroke} fill="rgba(5, 150, 105, 0.08)" stroke="#059669"
            strokeWidth={stroke} strokeDasharray={`${3 * stroke} ${2 * stroke}`} className="pointer-events-none" />
        ))}
        {review && page.review.map(({ box }, i) => (
          <rect key={`r${i}`} {...rect(box, 3 * stroke)} rx={2 * stroke} fill="none" stroke="#f59e0b"
            strokeWidth={1.5 * stroke} className="pointer-events-none" />
        ))}
        {highlight && (
          <rect ref={highlightRef} {...rect(highlight, 5 * stroke)} rx={3 * stroke} fill="none" stroke="#2563eb"
            strokeWidth={2 * stroke} className="pointer-events-none animate-pulse" />
        )}
        {drag && (
          <rect x={Math.min(drag.start[0], drag.end[0])} y={Math.min(drag.start[1], drag.end[1])}
            width={Math.abs(drag.end[0] - drag.start[0])} height={Math.abs(drag.end[1] - drag.start[1])}
            fill="rgba(234, 88, 12, 0.15)" stroke="#ea580c" strokeWidth={stroke}
            strokeDasharray={`${3 * stroke} ${2 * stroke}`} className="pointer-events-none" />
        )}
      </svg>
    </div>
  );
}

/** Every page of the document, one below the other. */
export function DocumentViewer({ pages, imageUrl, mode, zoom, highlight, onClick, onBox, onVisiblePage }: Props) {
  const { t } = useTranslation();
  const scroller = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [fit, setFit] = useState(800);

  useLayoutEffect(() => {
    const element = scroller.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setFit(Math.max(320, element.clientWidth - 64)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const scrolled = () => {
    const element = scroller.current;
    if (!element) return;
    const line = element.scrollTop + element.clientHeight / 3;
    let visible = 0;
    pageRefs.current.forEach((ref, i) => { if (ref && ref.offsetTop <= line) visible = i; });
    onVisiblePage(visible);
  };

  const width = fit * zoom;
  return (
    <div ref={scroller} onScroll={scrolled} className="h-full overflow-auto bg-stone-100">
      <div className="mx-auto flex flex-col items-center gap-3 py-6" style={{ width: width + 2 }}>
        {pages.map((page, n) => (
          <div key={n} ref={(el) => { pageRefs.current[n] = el; }} className="flex flex-col items-center gap-3">
            {pages.length > 1 && (
              <span className="text-xs font-medium text-stone-400">{t('findings.page', { n: n + 1 })}</span>
            )}
            {page ? (
              <PageCanvas n={n} page={page} imageUrl={imageUrl(n)} width={width} review={mode === 'review'}
                highlight={highlight?.page === n ? highlight.box : null} onClick={onClick} onBox={onBox} />
            ) : (
              <div className="animate-pulse rounded-sm bg-white ring-1 ring-stone-200" style={{ width, aspectRatio: '210 / 297' }} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
