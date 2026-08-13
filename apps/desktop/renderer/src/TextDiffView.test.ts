import { mount } from '@vue/test-utils';
import { describe, expect, it } from 'vitest';
import TextDiffView from './TextDiffView.vue';

describe('text diff view', () => {
  it('按参考在上、识别在下展示逐字差异并省略相同行', () => {
    const wrapper = mount(TextDiffView, {
      props: {
        referenceText: '相同一\n你好，世界\n相同三',
        recognizedText: '相同一\n你号世界\n相同三',
      },
    });
    const reference = wrapper.get('[data-test="reference-diff-side"]');
    const recognized = wrapper.get('[data-test="recognized-diff-side"]');

    expect(reference.element.compareDocumentPosition(recognized.element) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(reference.text()).toContain('你好，世界');
    expect(recognized.text()).toContain('你号世界');
    expect(wrapper.text()).not.toContain('相同一');
    expect(wrapper.text()).not.toContain('相同三');
    expect(wrapper.get('[data-test="reference-change"]').text()).toBe('好，');
    expect(wrapper.get('[data-test="recognized-change"]').text()).toBe('号');
  });

  it('使用 contracts 中的软件级默认差异色', () => {
    const wrapper = mount(TextDiffView, {
      props: {
        referenceText: '参考',
        recognizedText: '识别',
      },
    });
    const style = wrapper.get<HTMLElement>('[data-test="text-diff"]').element.style;

    expect(style.getPropertyValue('--speech-reference-diff-color')).toBe('#2E7D55');
    expect(style.getPropertyValue('--speech-recognized-diff-color')).toBe('#F72D5B');
  });

  it('完全一致时不渲染差异行', () => {
    const wrapper = mount(TextDiffView, {
      props: {
        referenceText: '完全一致',
        recognizedText: '完全一致',
      },
    });

    expect(wrapper.get('[data-test="text-diff-equal"]').text()).toContain('全文一致');
    expect(wrapper.find('[data-test="text-diff-hunk"]').exists()).toBe(false);
  });

  it('null 转写保持等待状态，空字符串作为真实差异处理', async () => {
    const wrapper = mount(TextDiffView, {
      props: {
        referenceText: '参考文本',
        recognizedText: null,
      },
    });

    expect(wrapper.get('[data-test="text-diff-pending"]').text()).toContain('尚未产生识别全文');
    expect(wrapper.find('[data-test="text-diff-hunk"]').exists()).toBe(false);

    await wrapper.setProps({ recognizedText: '' });

    expect(wrapper.find('[data-test="text-diff-pending"]').exists()).toBe(false);
    expect(wrapper.get('[data-test="recognized-diff-side"]').text()).toContain('∅ 无对应内容');
  });
});
