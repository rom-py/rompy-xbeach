# XBeach runtime image for rompy-xbeach.
#
# Builds XBeach from the Deltares SVN trunk at a pinned revision, with NetCDF output.
# Two binaries are built: a serial one and an MPI one (XBeach's MPI build refuses to
# run on a single process). The `xbeach` command is a wrapper that runs the MPI binary
# when launched by mpirun with 2 or more processes and the serial binary otherwise,
# so the same command works in both cases:
#
#   docker run --rm -v "$PWD":/data ghcr.io/rom-py/xbeach xbeach
#   docker run --rm -v "$PWD":/data ghcr.io/rom-py/xbeach mpirun -n 4 xbeach
#
# Build:
#   docker build -t xbeach docker
#   docker build -t xbeach --build-arg XBEACH_REVISION=6147 docker
#
# No ENTRYPOINT is set so the image works with rompy's DockerConfig backend, which
# runs `bash -c "<command>"` inside the container.

FROM ubuntu:24.04 AS builder

ARG XBEACH_REVISION=6147
ARG XBEACH_SVN=https://svn.oss.deltares.nl/repos/xbeach/trunk

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        ca-certificates \
        gcc \
        gfortran \
        libnetcdf-dev \
        libnetcdff-dev \
        libopenmpi-dev \
        libtool \
        make \
        openmpi-bin \
        pkg-config \
        python3 \
        python3-mako \
        subversion \
    && rm -rf /var/lib/apt/lists/*

RUN svn export --quiet --non-interactive -r "${XBEACH_REVISION}" "${XBEACH_SVN}" /opt/xbeach
WORKDIR /opt/xbeach

# The build scripts call `python`. The mako-generated .inc files are created up front
# and make runs serially because the Makefile lacks dependencies on the generated
# files and on Fortran module order, which breaks parallel builds. The library is
# linked statically so the serial and MPI binaries do not clash at runtime.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ./autogen.sh \
    && (cd src/xbeachlibrary && ../../scripts/generate.py)

RUN mkdir build-serial && cd build-serial \
    && FCFLAGS="-I/usr/include -O2" ../configure --with-netcdf --without-mpi \
        --disable-shared --prefix=/opt/xbeach/serial \
    && make && make install

RUN mkdir build-mpi && cd build-mpi \
    && FCFLAGS="-I/usr/include -O2" ../configure --with-netcdf --with-mpi \
        --disable-shared --prefix=/opt/xbeach/mpi \
    && make && make install

FROM ubuntu:24.04

ARG XBEACH_REVISION=6147

LABEL org.opencontainers.image.title="xbeach" \
      org.opencontainers.image.description="XBeach (SVN trunk r${XBEACH_REVISION}) with NetCDF and MPI, for use with rompy-xbeach" \
      org.opencontainers.image.source="https://github.com/rom-py/rompy-xbeach" \
      org.opencontainers.image.version="r${XBEACH_REVISION}" \
      org.opencontainers.image.licenses="LGPL-3.0-or-later"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgfortran5 \
        libnetcdf19t64 \
        libnetcdff7 \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/xbeach/serial/bin/xbeach /usr/local/bin/xbeach-serial
COPY --from=builder /opt/xbeach/mpi/bin/xbeach /usr/local/bin/xbeach-mpi
COPY xbeach.sh /usr/local/bin/xbeach

# Containers usually run as root, which Open MPI refuses by default
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    XBEACH_REVISION=${XBEACH_REVISION}

WORKDIR /data
CMD ["xbeach"]
