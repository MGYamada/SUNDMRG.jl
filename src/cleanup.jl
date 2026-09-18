# Run cleanup exactly once. Keep a calculation's exception and backtrace when
# cleanup succeeds, and retain both failures when cleanup also throws.
function _with_cleanup(f, cleanup)
    result = try
        f()
    catch error
        primary = CapturedException(error, catch_backtrace())
        try
            cleanup()
        catch cleanup_error
            cleanup_backtrace = catch_backtrace()
            # CompositeException's default display abbreviates later failures.
            # Report the cleanup details as well as retaining both exceptions.
            @error "Cleanup failed while handling an earlier exception" exception = (cleanup_error, cleanup_backtrace)
            throw(CompositeException(Any[primary, CapturedException(cleanup_error, cleanup_backtrace)]))
        end
        rethrow()
    end
    cleanup()
    return result
end
