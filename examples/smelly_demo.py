"""Demo file with a few classic smells."""
import time


def process_data2(x, y):
    t = 0
    for i in range(len(x)):
        d = x[i]
        if d > 100:
            t = t + d * 2
    tmp = t / 3
    if tmp > 50:
        return tmp + 5
    return tmp


def do_stuff(req):
    data = req.get("payload", [])
    out = []
    for item in data:
        status = item.get("status", 0)
        if status == 1:
            out.append(item)
        elif status == 2:
            out.append(item)
        elif status == 3:
            out.append(item)
        elif status == 4:
            out.append(item)
        elif status == 5:
            out.append(item)
        elif status == 6:
            out.append(item)
        else:
            out.append(item)
    time.sleep(2)
    return out


def save_and_load_and_mail_and_log_and_notify(records, target, sender, channel, tls, timeout, retries, backoff):
    saved = []
    for r in records:
        if r is not None:
            saved.append(r)
    return saved


def normalize_metric(raw):
    """Small clean-ish helper for contrast."""
    clean = [x.strip().lower() for x in (raw or [])]
    return sorted(clean)
