import { describe, it, expect, vi } from 'vitest';

const post = vi.fn();
vi.mock('../api/client', () => ({ api: { post: (...a: any[]) => post(...a) } }));
vi.mock('react', async () => ({ ...(await vi.importActual<any>('react')), useCallback: (f: any) => f }));

import { useSearchApi } from '../api/useSearchApi';
import { useSearchStore } from '../store/useSearchStore';

describe('select-face không gửi trùng', () => {
  it('hai lần gọi trùng khi request trước còn chạy → 1 request', async () => {
    let resolve: (v: any) => void = () => {};
    post.mockReturnValueOnce(new Promise((r) => { resolve = r; }));
    useSearchStore.setState({ sessionId: 's1' });
    const { selectFace } = useSearchApi();
    const a = selectFace('s1', 0);
    const b = selectFace('s1', 0);
    resolve({ data: { warnings: [], cropped_preview_url: '/x.png' } });
    await Promise.all([a, b]);
    expect(post).toHaveBeenCalledTimes(1);

    post.mockResolvedValueOnce({ data: { warnings: [], cropped_preview_url: '/y.png' } });
    await selectFace('s1', 0);                      // request trước đã xong → gửi mới
    expect(post).toHaveBeenCalledTimes(2);
  });
});
