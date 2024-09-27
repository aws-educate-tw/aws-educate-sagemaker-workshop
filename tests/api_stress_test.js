import http from 'k6/http';
import {
    check
} from 'k6';

export const options = {
    vus: 500, // 並發的虛擬使用者數量
    duration: '1m', // 測試持續 1 分鐘
};

export default function () {
    const url = 'YOUR_API_ENDPOINT';
    const payload = JSON.stringify({
        "model": "psy-1",
        "system": "",
        "messages": [{
            "role": "user",
            "content": "能迅速掌握新技能，並根據工作需求快速調整自己的步伐。你在面對高強度的工作壓力時，能夠全力以赴，展現出卓越的高效率。無論是處理複雜的項目還是應對緊急情況，你總是能夠迎刃而解。"
        }],
        "max_tokens": 1024,
        "temperature": 0.5,
        "stream": true
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const res = http.post(url, payload, params);

    // 檢查是否狀態碼為 200
    check(res, {
        'status was 200': (r) => r.status === 200,
    });
}