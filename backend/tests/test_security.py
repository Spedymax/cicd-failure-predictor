from app.core.security import compute_signature, verify_signature


def test_signature_roundtrip():
    body = b'{"foo":"bar"}'
    sig = compute_signature("secret", body)
    assert sig.startswith("sha256=")
    assert verify_signature("secret", body, sig)


def test_rejects_wrong_signature():
    body = b'{"foo":"bar"}'
    sig = compute_signature("secret", body)
    assert not verify_signature("other-secret", body, sig)


def test_rejects_missing_header():
    assert not verify_signature("secret", b"{}", None)


def test_rejects_tampered_body():
    body = b'{"foo":"bar"}'
    sig = compute_signature("secret", body)
    assert not verify_signature("secret", b'{"foo":"baz"}', sig)
