const list = document.querySelector('#recommendation-list');
const status = document.querySelector('#recommendation-status');

if (list && status) {
  list.addEventListener('submit', async (event) => {
    const form = event.target.closest('form[data-async-rating]');
    if (!form) return;
    event.preventDefault();
    const select = form.querySelector('select[name="rating"]');
    if (!select.value) return;
    const button = form.querySelector('button[type="submit"]');
    button.disabled = true;
    status.textContent = 'Đang lưu đánh giá và tải lại gợi ý…';
    try {
      const token = document.querySelector('meta[name="csrf-token"]').content;
      const save = await fetch(`/api/ratings/${encodeURIComponent(form.dataset.itemId)}`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': token },
        body: JSON.stringify({ rating: Number(select.value) }),
      });
      if (!save.ok) throw new Error('Không lưu được đánh giá.');
      const response = await fetch(`/api/recommendations?method=${encodeURIComponent(list.dataset.method)}`, {
        credentials: 'same-origin',
      });
      if (!response.ok) throw new Error('Không tải được gợi ý mới.');
      const items = await response.json();
      list.replaceChildren(...items.map((item, index) => recommendationCard(item, index, token)));
      status.textContent = `Đã cập nhật đánh giá. Đang hiển thị ${items.length} gợi ý${items.length < 10 ? '; danh mục hiện không còn đủ 10 sách hợp lệ' : ''}.`;
    } catch (error) {
      status.textContent = `${error.message} Hãy tải lại trang và thử lại.`;
      button.disabled = false;
    }
  });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function recommendationCard(item, index, token) {
  const card = element('li', 'recommendation-card');
  card.append(element('span', 'rank', String(index + 1).padStart(2, '0')));
  const content = element('div');
  const title = element('h2');
  const link = element('a', '', item.title);
  link.href = `/books/${encodeURIComponent(item.canonical_item_id)}`;
  title.append(link);
  content.append(title);
  content.append(element('p', 'muted', `${item.authors.join(', ') || 'Tác giả chưa rõ'}${item.genres.length ? ' · ' + item.genres.join(', ') : ''}`));
  content.append(element('p', '', item.description ? `${item.description.slice(0, 170)}${item.description.length > 170 ? '…' : ''}` : 'Chưa có mô tả.'));
  content.append(element('p', 'reason', `Lý do: ${item.reason}`));
  if (item.source === 'popularity_fallback') content.append(element('p', 'muted', 'Gợi ý dự phòng theo độ phổ biến trong tập huấn luyện.'));
  card.append(content);
  const form = element('form', 'rating-inline');
  form.method = 'post';
  form.action = `/ratings/${encodeURIComponent(item.canonical_item_id)}`;
  form.dataset.asyncRating = '';
  form.dataset.itemId = item.canonical_item_id;
  for (const [name, value] of [['csrf_token', token], ['next', location.pathname + location.search]]) {
    const hidden = element('input');
    hidden.type = 'hidden';
    hidden.name = name;
    hidden.value = value;
    form.append(hidden);
  }
  const label = element('label', '', 'Đánh giá');
  const select = element('select', 'rating-select');
  select.name = 'rating';
  select.required = true;
  const placeholder = element('option', '', 'Chọn điểm');
  placeholder.value = '';
  placeholder.disabled = true;
  placeholder.selected = true;
  select.append(placeholder);
  for (let rating = 1; rating <= 5; rating += 1) {
    const option = element('option', '', String(rating));
    option.value = String(rating);
    select.append(option);
  }
  label.append(select);
  form.append(label);
  const button = element('button', 'button', 'Lưu');
  button.type = 'submit';
  form.append(button);
  card.append(form);
  return card;
}
