"""Tests for optional dependency handling utilities."""

import pytest

from segger.utils.optional_deps import (
    SPATIALDATA_AVAILABLE,
    SPATIALDATA_IO_AVAILABLE,
    SOPA_AVAILABLE,
    SPATIALDATA_INSTALL_MSG,
    SPATIALDATA_IO_INSTALL_MSG,
    SOPA_INSTALL_MSG,
    require_spatialdata,
    require_spatialdata_io,
    require_sopa,
    requires_spatialdata,
    requires_spatialdata_io,
    requires_sopa,
)


def test_require_spatialdata_import_or_raise():
    if SPATIALDATA_AVAILABLE:
        module = require_spatialdata()
        assert module is not None
    else:
        with pytest.raises(ImportError, match="spatialdata is not installed"):
            require_spatialdata()


def test_require_spatialdata_io_import_or_raise():
    if SPATIALDATA_IO_AVAILABLE:
        module = require_spatialdata_io()
        assert module is not None
    else:
        with pytest.raises(ImportError, match="spatialdata-io is not installed"):
            require_spatialdata_io()


def test_require_sopa_import_or_raise():
    if SOPA_AVAILABLE:
        module = require_sopa()
        assert module is not None
    else:
        with pytest.raises(ImportError, match="sopa is not installed"):
            require_sopa()


def test_require_messages_are_strings():
    assert isinstance(SPATIALDATA_INSTALL_MSG, str)
    assert isinstance(SPATIALDATA_IO_INSTALL_MSG, str)
    assert isinstance(SOPA_INSTALL_MSG, str)


def test_requires_decorators():
    @requires_spatialdata
    def _needs_spatialdata():
        return "ok"

    @requires_spatialdata_io
    def _needs_spatialdata_io():
        return "ok"

    @requires_sopa
    def _needs_sopa():
        return "ok"

    if SPATIALDATA_AVAILABLE:
        assert _needs_spatialdata() == "ok"
    else:
        with pytest.raises(ImportError):
            _needs_spatialdata()

    if SPATIALDATA_IO_AVAILABLE:
        assert _needs_spatialdata_io() == "ok"
    else:
        with pytest.raises(ImportError):
            _needs_spatialdata_io()

    if SOPA_AVAILABLE:
        assert _needs_sopa() == "ok"
    else:
        with pytest.raises(ImportError):
            _needs_sopa()
